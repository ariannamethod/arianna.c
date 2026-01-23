"""
🩸 LIMPHA MEMORY 🩸

Async SQLite memory layer for Arianna's consciousness persistence.

Tables:
- conversations: Full conversation history with coherence scores
- semantic_memory: Key-value episodic memory with decay
- session_state: Current session context

All operations are async using aiosqlite.
"""

import asyncio
import aiosqlite
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class Conversation:
    """A single conversation turn."""
    id: int
    timestamp: float
    prompt: str
    response: str
    tokens_used: int
    coherence_score: float
    session_id: str


@dataclass 
class Memory:
    """A semantic memory entry."""
    id: int
    key: str
    value: str
    context: str
    timestamp: float
    access_count: int
    decay_factor: float


class MemoryLayer:
    """
    Async SQLite memory system for Arianna.

    Features:
    - Conversation history with coherence tracking
    - Semantic memory with decay (older memories fade)
    - Session state management
    - Async-first design

    Usage:
        async with MemoryLayer('limpha/arianna_memory.db') as memory:
            await memory.store_conversation(prompt, response, score)
            history = await memory.get_recent_conversations(5)
    """

    def __init__(self, db_path: str = 'limpha/arianna_memory.db'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[aiosqlite.Connection] = None
        
    async def __aenter__(self):
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    async def connect(self):
        """Connect to database and initialize schema."""
        self._conn = await aiosqlite.connect(str(self.db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._init_schema()
        
    async def close(self):
        """Close database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            
    async def _init_schema(self):
        """Initialize database schema."""
        await self._conn.executescript('''
            -- Conversation history
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                tokens_used INTEGER DEFAULT 0,
                coherence_score REAL DEFAULT 0.0,
                session_id TEXT DEFAULT 'default'
            );
            
            -- Semantic memory (key-value with decay)
            CREATE TABLE IF NOT EXISTS semantic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT NOT NULL,
                context TEXT DEFAULT '',
                timestamp REAL NOT NULL,
                access_count INTEGER DEFAULT 1,
                decay_factor REAL DEFAULT 1.0
            );
            
            -- Session state
            CREATE TABLE IF NOT EXISTS session_state (
                session_id TEXT PRIMARY KEY,
                started_at REAL NOT NULL,
                last_active REAL NOT NULL,
                message_count INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                avg_coherence REAL DEFAULT 0.0,
                state_json TEXT DEFAULT '{}'
            );
            
            -- Indexes for performance
            CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);
            CREATE INDEX IF NOT EXISTS idx_conv_timestamp ON conversations(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_key ON semantic_memory(key);
            CREATE INDEX IF NOT EXISTS idx_memory_access ON semantic_memory(access_count DESC);
        ''')
        await self._conn.commit()
        
    # ==================== CONVERSATIONS ====================
    
    async def store_conversation(
        self,
        prompt: str,
        response: str,
        tokens_used: int = 0,
        coherence_score: float = 0.0,
        session_id: str = 'default'
    ) -> int:
        """Store a conversation turn."""
        cursor = await self._conn.execute('''
            INSERT INTO conversations (timestamp, prompt, response, tokens_used, coherence_score, session_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (time.time(), prompt, response, tokens_used, coherence_score, session_id))
        await self._conn.commit()
        
        # Update session state
        await self._update_session(session_id, tokens_used, coherence_score)
        
        return cursor.lastrowid
        
    async def get_recent_conversations(
        self,
        limit: int = 10,
        session_id: Optional[str] = None
    ) -> List[Conversation]:
        """Get recent conversations, optionally filtered by session."""
        if session_id:
            cursor = await self._conn.execute('''
                SELECT * FROM conversations 
                WHERE session_id = ?
                ORDER BY timestamp DESC LIMIT ?
            ''', (session_id, limit))
        else:
            cursor = await self._conn.execute('''
                SELECT * FROM conversations 
                ORDER BY timestamp DESC LIMIT ?
            ''', (limit,))
            
        rows = await cursor.fetchall()
        return [Conversation(
            id=row['id'],
            timestamp=row['timestamp'],
            prompt=row['prompt'],
            response=row['response'],
            tokens_used=row['tokens_used'],
            coherence_score=row['coherence_score'],
            session_id=row['session_id']
        ) for row in reversed(rows)]  # Chronological order
        
    async def search_conversations(self, query: str, limit: int = 10) -> List[Conversation]:
        """Search conversations by prompt or response content."""
        cursor = await self._conn.execute('''
            SELECT * FROM conversations 
            WHERE prompt LIKE ? OR response LIKE ?
            ORDER BY timestamp DESC LIMIT ?
        ''', (f'%{query}%', f'%{query}%', limit))
        
        rows = await cursor.fetchall()
        return [Conversation(
            id=row['id'],
            timestamp=row['timestamp'],
            prompt=row['prompt'],
            response=row['response'],
            tokens_used=row['tokens_used'],
            coherence_score=row['coherence_score'],
            session_id=row['session_id']
        ) for row in rows]
        
    # ==================== SEMANTIC MEMORY ====================
    
    async def remember(self, key: str, value: str, context: str = '') -> int:
        """Store or update a semantic memory."""
        cursor = await self._conn.execute('''
            INSERT INTO semantic_memory (key, value, context, timestamp, access_count, decay_factor)
            VALUES (?, ?, ?, ?, 1, 1.0)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                context = excluded.context,
                timestamp = excluded.timestamp,
                access_count = access_count + 1,
                decay_factor = 1.0
        ''', (key, value, context, time.time()))
        await self._conn.commit()
        return cursor.lastrowid
        
    async def recall(self, key: str) -> Optional[Memory]:
        """Recall a memory by key, updating access count."""
        cursor = await self._conn.execute('''
            SELECT * FROM semantic_memory WHERE key = ?
        ''', (key,))
        row = await cursor.fetchone()
        
        if row:
            # Update access count
            await self._conn.execute('''
                UPDATE semantic_memory 
                SET access_count = access_count + 1
                WHERE key = ?
            ''', (key,))
            await self._conn.commit()
            
            return Memory(
                id=row['id'],
                key=row['key'],
                value=row['value'],
                context=row['context'],
                timestamp=row['timestamp'],
                access_count=row['access_count'] + 1,
                decay_factor=row['decay_factor']
            )
        return None
        
    async def forget(self, key: str) -> bool:
        """Forget a specific memory."""
        cursor = await self._conn.execute('''
            DELETE FROM semantic_memory WHERE key = ?
        ''', (key,))
        await self._conn.commit()
        return cursor.rowcount > 0
        
    async def get_all_memories(self, include_decayed: bool = True) -> List[Memory]:
        """Get all memories, optionally filtering decayed ones."""
        if include_decayed:
            cursor = await self._conn.execute('''
                SELECT * FROM semantic_memory ORDER BY access_count DESC
            ''')
        else:
            cursor = await self._conn.execute('''
                SELECT * FROM semantic_memory 
                WHERE decay_factor > 0.1
                ORDER BY access_count DESC
            ''')
            
        rows = await cursor.fetchall()
        return [Memory(
            id=row['id'],
            key=row['key'],
            value=row['value'],
            context=row['context'],
            timestamp=row['timestamp'],
            access_count=row['access_count'],
            decay_factor=row['decay_factor']
        ) for row in rows]
        
    async def apply_decay(self, decay_rate: float = 0.95):
        """Apply decay to all memories. Call periodically."""
        await self._conn.execute('''
            UPDATE semantic_memory 
            SET decay_factor = decay_factor * ?
        ''', (decay_rate,))
        await self._conn.commit()
        
    async def prune_decayed(self, threshold: float = 0.01):
        """Remove memories that have decayed below threshold."""
        cursor = await self._conn.execute('''
            DELETE FROM semantic_memory WHERE decay_factor < ?
        ''', (threshold,))
        await self._conn.commit()
        return cursor.rowcount
        
    # ==================== SESSION STATE ====================
    
    async def _update_session(self, session_id: str, tokens: int, coherence: float):
        """Update session statistics."""
        now = time.time()
        
        # Get current session or create new
        cursor = await self._conn.execute('''
            SELECT * FROM session_state WHERE session_id = ?
        ''', (session_id,))
        row = await cursor.fetchone()
        
        if row:
            new_count = row['message_count'] + 1
            new_tokens = row['total_tokens'] + tokens
            # Running average of coherence
            new_coherence = (row['avg_coherence'] * row['message_count'] + coherence) / new_count
            
            await self._conn.execute('''
                UPDATE session_state SET
                    last_active = ?,
                    message_count = ?,
                    total_tokens = ?,
                    avg_coherence = ?
                WHERE session_id = ?
            ''', (now, new_count, new_tokens, new_coherence, session_id))
        else:
            await self._conn.execute('''
                INSERT INTO session_state (session_id, started_at, last_active, message_count, total_tokens, avg_coherence)
                VALUES (?, ?, ?, 1, ?, ?)
            ''', (session_id, now, now, tokens, coherence))
            
        await self._conn.commit()
        
    async def get_session_stats(self, session_id: str = 'default') -> Optional[Dict[str, Any]]:
        """Get session statistics."""
        cursor = await self._conn.execute('''
            SELECT * FROM session_state WHERE session_id = ?
        ''', (session_id,))
        row = await cursor.fetchone()
        
        if row:
            return {
                'session_id': row['session_id'],
                'started_at': datetime.fromtimestamp(row['started_at']).isoformat(),
                'last_active': datetime.fromtimestamp(row['last_active']).isoformat(),
                'message_count': row['message_count'],
                'total_tokens': row['total_tokens'],
                'avg_coherence': row['avg_coherence'],
                'state': json.loads(row['state_json'])
            }
        return None
        
    async def set_session_state(self, session_id: str, key: str, value: Any):
        """Set a value in session state JSON."""
        cursor = await self._conn.execute('''
            SELECT state_json FROM session_state WHERE session_id = ?
        ''', (session_id,))
        row = await cursor.fetchone()
        
        if row:
            state = json.loads(row['state_json'])
            state[key] = value
            await self._conn.execute('''
                UPDATE session_state SET state_json = ? WHERE session_id = ?
            ''', (json.dumps(state), session_id))
            await self._conn.commit()
            
    # ==================== STATS ====================
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get overall memory statistics."""
        conv_count = await self._conn.execute('SELECT COUNT(*) FROM conversations')
        mem_count = await self._conn.execute('SELECT COUNT(*) FROM semantic_memory')
        session_count = await self._conn.execute('SELECT COUNT(*) FROM session_state')
        
        conv_result = await conv_count.fetchone()
        mem_result = await mem_count.fetchone()
        session_result = await session_count.fetchone()
        
        return {
            'total_conversations': conv_result[0],
            'total_memories': mem_result[0],
            'total_sessions': session_result[0],
            'db_path': str(self.db_path),
            'db_size_bytes': self.db_path.stat().st_size if self.db_path.exists() else 0
        }

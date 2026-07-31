package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const bodyInventorySchema = "arianna.body_inventory.v1"

type bodyInventoryOrganSpec struct {
	Name     string
	Role     string
	Kind     string
	Path     string
	Required bool
}

type bodyInventoryOrgan struct {
	Name     string `json:"name"`
	Role     string `json:"role"`
	Kind     string `json:"kind"`
	Path     string `json:"path"`
	Required bool   `json:"required"`
	Present  bool   `json:"present"`
	Size     int64  `json:"size,omitempty"`
	Reason   string `json:"reason,omitempty"`
}

type bodyInventoryRouteSpec struct {
	Route          string
	Backend        string
	Entrypoint     string
	RequiredOrgans []string
	AnyOfOrgans    []string
}

type bodyInventoryRouteAvailability struct {
	Route          string   `json:"route"`
	Backend        string   `json:"backend,omitempty"`
	Entrypoint     string   `json:"entrypoint,omitempty"`
	RequiredOrgans []string `json:"required_organs,omitempty"`
	AnyOfOrgans    []string `json:"any_of_organs,omitempty"`
	MissingOrgans  []string `json:"missing_organs,omitempty"`
	Available      bool     `json:"available"`
	Reason         string   `json:"reason,omitempty"`
}

type bodyInventoryReceipt struct {
	Schema            string                           `json:"schema"`
	Mode              string                           `json:"mode"`
	Root              string                           `json:"root"`
	Status            string                           `json:"status"`
	CoreReady         bool                             `json:"core_ready"`
	OptionalReady     bool                             `json:"optional_ready"`
	LiveTrioAllowed   bool                             `json:"live_trio_allowed"`
	ContinueAllowed   bool                             `json:"continue_allowed"`
	DegradedMode      bool                             `json:"degraded_mode"`
	MutatesState      bool                             `json:"mutates_state"`
	RequiredMissing   []string                         `json:"required_missing,omitempty"`
	OptionalMissing   []string                         `json:"optional_missing,omitempty"`
	Organs            []bodyInventoryOrgan             `json:"organs"`
	RouteAvailability []bodyInventoryRouteAvailability `json:"route_availability"`
	Reason            string                           `json:"reason"`
}

func bodyInventorySpecs() []bodyInventoryOrganSpec {
	janusModel := getenvDefault("A2A_JANUS_MODEL", "weights/arianna_v4_sft_f16.gguf")
	resonanceModel := getenvDefault("A2A_RESONANCE_MODEL", "weights/arianna_resonance_v3_f16.gguf")
	nanoModel := getenvDefault("A2A_NANO_MODEL", "weights/nano_arianna_f16.gguf")
	return []bodyInventoryOrganSpec{
		{Name: "janus-binary", Role: "external mouth daemon", Kind: "binary", Path: "./arianna", Required: true},
		{Name: "janus-weight", Role: "external mouth weight", Kind: "weight", Path: janusModel, Required: true},
		{Name: "resonance-binary", Role: "inner-world daemon", Kind: "binary", Path: "./arianna_resonance", Required: true},
		{Name: "resonance-weight", Role: "inner-world weight", Kind: "weight", Path: resonanceModel, Required: true},
		{Name: "nano-binary", Role: "subconscious one-shot engine", Kind: "binary", Path: "./nano-arianna", Required: false},
		{Name: "nano-weight", Role: "subconscious weight", Kind: "weight", Path: nanoModel, Required: false},
		{Name: "doe-binary", Role: "notorch-native nano parliament engine", Kind: "binary", Path: "./doe_field", Required: false},
		{Name: "chorus-binary", Role: "polyphonic nano substrate", Kind: "binary", Path: "./chorus-arianna", Required: false},
		{Name: "kk-binary", Role: "Knowledge Kernel retriever", Kind: "binary", Path: "./kk-cli", Required: false},
		{Name: "kk-db", Role: "Knowledge Kernel dream memory", Kind: "state", Path: "weights/nano.kk.db", Required: false},
	}
}

func bodyInventoryRouteSpecs() []bodyInventoryRouteSpec {
	return []bodyInventoryRouteSpec{
		{
			Route:          "direct",
			Backend:        "nano",
			Entrypoint:     "direct",
			RequiredOrgans: []string{"nano-weight"},
			AnyOfOrgans:    []string{"nano-binary", "doe-binary"},
		},
		{
			Route:          "chorus",
			Backend:        "chorus-arianna",
			Entrypoint:     "field",
			RequiredOrgans: []string{"chorus-binary", "nano-weight"},
		},
		{
			Route:          "qloop",
			Backend:        "chorus-arianna",
			Entrypoint:     "qloop",
			RequiredOrgans: []string{"chorus-binary", "nano-weight"},
		},
		{
			Route:          "qloop_hint_qa",
			Backend:        "chorus-arianna",
			Entrypoint:     "qloop_hint_qa",
			RequiredOrgans: []string{"chorus-binary", "nano-weight"},
		},
		{
			Route:          "qloop_target",
			Backend:        "chorus-arianna",
			Entrypoint:     "qloop_target",
			RequiredOrgans: []string{"chorus-binary", "nano-weight"},
		},
		{
			Route:          "user_bridge",
			Backend:        "chorus-arianna",
			Entrypoint:     "repl_user_bridge",
			RequiredOrgans: []string{"chorus-binary", "nano-weight"},
		},
	}
}

func getenvDefault(name, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(name)); v != "" {
		return v
	}
	return fallback
}

func bodyInventoryRoot() string {
	root := strings.TrimSpace(os.Getenv("AM_BODY_INVENTORY_ROOT"))
	if root == "" {
		root = "."
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return root
	}
	return abs
}

func resolveInventoryPath(root, path string) string {
	if filepath.IsAbs(path) {
		return path
	}
	if strings.HasPrefix(path, "./") {
		path = strings.TrimPrefix(path, "./")
	}
	return filepath.Join(root, path)
}

func inspectBodyOrgan(root string, spec bodyInventoryOrganSpec) bodyInventoryOrgan {
	path := resolveInventoryPath(root, spec.Path)
	organ := bodyInventoryOrgan{
		Name:     spec.Name,
		Role:     spec.Role,
		Kind:     spec.Kind,
		Path:     path,
		Required: spec.Required,
	}
	info, err := os.Stat(path)
	if err != nil {
		organ.Reason = "missing_" + spec.Name
		return organ
	}
	if info.IsDir() {
		organ.Reason = "not_a_file_" + spec.Name
		return organ
	}
	if spec.Kind == "binary" && info.Mode()&0111 == 0 {
		organ.Size = info.Size()
		organ.Reason = "not_executable_" + spec.Name
		return organ
	}
	if (spec.Kind == "weight" || spec.Kind == "state") && info.Size() <= 0 {
		organ.Reason = "empty_" + spec.Name
		return organ
	}
	organ.Present = true
	organ.Size = info.Size()
	return organ
}

func inspectBodyInventory(root string) bodyInventoryReceipt {
	if root == "" {
		root = bodyInventoryRoot()
	}
	receipt := bodyInventoryReceipt{
		Schema:          bodyInventorySchema,
		Mode:            "live_trio",
		Root:            root,
		CoreReady:       true,
		OptionalReady:   true,
		ContinueAllowed: true,
		MutatesState:    false,
	}
	for _, spec := range bodyInventorySpecs() {
		organ := inspectBodyOrgan(root, spec)
		receipt.Organs = append(receipt.Organs, organ)
		if organ.Present {
			continue
		}
		if organ.Required {
			receipt.CoreReady = false
			receipt.RequiredMissing = append(receipt.RequiredMissing, spec.Name)
		} else {
			receipt.OptionalReady = false
			receipt.OptionalMissing = append(receipt.OptionalMissing, spec.Name)
		}
	}
	sort.Strings(receipt.RequiredMissing)
	sort.Strings(receipt.OptionalMissing)
	receipt.RouteAvailability = inspectBodyRouteAvailability(receipt)
	receipt.LiveTrioAllowed = receipt.CoreReady
	receipt.DegradedMode = !receipt.CoreReady || !receipt.OptionalReady
	switch {
	case !receipt.CoreReady:
		receipt.Status = "blocked"
		receipt.Reason = "required live-trio organs missing; keep process-level inspection alive, but do not start the live trio"
	case !receipt.OptionalReady:
		receipt.Status = "degraded"
		receipt.Reason = "optional organs missing; run the available body and record the degraded route"
	default:
		receipt.Status = "ready"
		receipt.Reason = "all declared body organs present"
	}
	return receipt
}

func inspectBodyRouteAvailability(receipt bodyInventoryReceipt) []bodyInventoryRouteAvailability {
	routes := make([]bodyInventoryRouteAvailability, 0, len(bodyInventoryRouteSpecs()))
	for _, spec := range bodyInventoryRouteSpecs() {
		route := bodyInventoryRouteAvailability{
			Route:          spec.Route,
			Backend:        spec.Backend,
			Entrypoint:     spec.Entrypoint,
			RequiredOrgans: append([]string(nil), spec.RequiredOrgans...),
			AnyOfOrgans:    append([]string(nil), spec.AnyOfOrgans...),
			Available:      true,
		}
		missing := make([]string, 0, len(spec.RequiredOrgans)+len(spec.AnyOfOrgans))
		for _, organ := range spec.RequiredOrgans {
			if !receipt.organPresent(organ) {
				missing = append(missing, organ)
			}
		}
		if len(spec.AnyOfOrgans) > 0 {
			anyPresent := false
			for _, organ := range spec.AnyOfOrgans {
				if receipt.organPresent(organ) {
					anyPresent = true
					break
				}
			}
			if !anyPresent {
				missing = append(missing, spec.AnyOfOrgans...)
			}
		}
		if len(missing) > 0 {
			sort.Strings(missing)
			route.Available = false
			route.MissingOrgans = missing
			route.Reason = "missing_route_organs:" + strings.Join(missing, ",")
		}
		routes = append(routes, route)
	}
	return routes
}

func (receipt bodyInventoryReceipt) organPresent(name string) bool {
	for _, organ := range receipt.Organs {
		if organ.Name == name {
			return organ.Present
		}
	}
	return false
}

func (receipt bodyInventoryReceipt) routeAvailability(route string) (bodyInventoryRouteAvailability, bool) {
	for _, availability := range receipt.RouteAvailability {
		if availability.Route == route {
			return availability, true
		}
	}
	return bodyInventoryRouteAvailability{}, false
}

func (receipt bodyInventoryReceipt) routeAvailable(route string) bool {
	availability, ok := receipt.routeAvailability(route)
	return ok && availability.Available
}

func requireBodyInventoryLiveTrio(receipt bodyInventoryReceipt) error {
	if receipt.LiveTrioAllowed {
		return nil
	}
	if len(receipt.RequiredMissing) == 0 {
		return fmt.Errorf("body inventory blocked: live trio not allowed")
	}
	return fmt.Errorf("body inventory blocked: required organs missing: %s", strings.Join(receipt.RequiredMissing, ","))
}

func writeBodyInventoryReceipt(path string, receipt bodyInventoryReceipt) error {
	if strings.TrimSpace(path) == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetEscapeHTML(false)
	return enc.Encode(receipt)
}

func runBodyInventorySmoke() error {
	receipt := inspectBodyInventory(bodyInventoryRoot())
	if err := writeBodyInventoryReceipt(os.Getenv("AM_BODY_INVENTORY_LOG"), receipt); err != nil {
		return err
	}
	raw, err := json.Marshal(receipt)
	if err != nil {
		return err
	}
	fmt.Println(string(raw))
	fmt.Printf("body-inventory: status=%s live_trio_allowed=%t degraded=%t required_missing=%d optional_missing=%d\n",
		receipt.Status, receipt.LiveTrioAllowed, receipt.DegradedMode, len(receipt.RequiredMissing), len(receipt.OptionalMissing))
	if os.Getenv("AM_BODY_INVENTORY_REQUIRE_CORE") == "1" && !receipt.CoreReady {
		return fmt.Errorf("required organs missing: %s", strings.Join(receipt.RequiredMissing, ","))
	}
	return nil
}

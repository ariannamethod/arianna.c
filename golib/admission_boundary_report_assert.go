package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func runAdmissionLiveRouteBoundaryReportAssert(args []string) error {
	if len(args) < 2 {
		return fmt.Errorf("usage: --admission-live-route-boundary-report-assert REPORT EXPECTED_RECEIPTS [STAGE...]")
	}
	report, root, err := readAdmissionLiveRouteBoundaryReportForAssert(args[0])
	if err != nil {
		return err
	}
	expectedReceipts, err := parseAdmissionLiveRouteBoundaryReportReceiptCount(args[1])
	if err != nil {
		return err
	}
	if err := admissionLiveRouteBoundaryReportSchemaError(report, root); err != nil {
		return err
	}
	if !report.Passed {
		return fmt.Errorf("boundary report did not pass")
	}
	if report.ReceiptsChecked != expectedReceipts {
		return fmt.Errorf("boundary report receipt count mismatch")
	}
	if !admissionLiveRouteBoundaryReportHasObject(root, "boundary") {
		return fmt.Errorf("boundary report projection missing")
	}
	for _, stageName := range args[2:] {
		if strings.TrimSpace(stageName) == "" {
			return fmt.Errorf("empty boundary report stage name")
		}
		stage, count := admissionLiveRouteBoundaryReportStageByName(report, stageName)
		if count == 0 {
			return fmt.Errorf("boundary report stage missing: %s", stageName)
		}
		if count > 1 {
			return fmt.Errorf("boundary report stage duplicated: %s", stageName)
		}
		if !stage.Passed {
			return fmt.Errorf("boundary report stage did not pass: %s", stageName)
		}
	}
	return nil
}

func runAdmissionLiveRouteBoundaryReportFailedDiagnosticsAssert(args []string) error {
	if len(args) < 3 {
		return fmt.Errorf("usage: --admission-live-route-boundary-report-failed-diagnostics-assert REPORT STAGE EXPECTED_MISMATCH [EXPECTED_MISMATCH...]")
	}
	report, root, err := readAdmissionLiveRouteBoundaryReportForAssert(args[0])
	if err != nil {
		return err
	}
	stageName := args[1]
	if strings.TrimSpace(stageName) == "" {
		return fmt.Errorf("boundary report stage name missing")
	}
	if err := admissionLiveRouteBoundaryReportSchemaError(report, root); err != nil {
		return err
	}
	if report.Passed {
		return fmt.Errorf("boundary report did not fail")
	}
	if !admissionLiveRouteBoundaryReportHasReason(report, "boundary_mismatch:"+stageName) {
		return fmt.Errorf("boundary mismatch reason missing: %s", stageName)
	}
	stage, count := admissionLiveRouteBoundaryReportStageByName(report, stageName)
	if count == 0 {
		return fmt.Errorf("boundary report stage missing: %s", stageName)
	}
	if count > 1 {
		return fmt.Errorf("boundary report stage duplicated: %s", stageName)
	}
	if stage.Passed {
		return fmt.Errorf("boundary report stage did not fail: %s", stageName)
	}
	if len(stage.Mismatches) == 0 {
		return fmt.Errorf("boundary report stage mismatches missing: %s", stageName)
	}
	mismatches := make(map[string]struct{}, len(stage.Mismatches))
	for _, mismatch := range stage.Mismatches {
		mismatches[mismatch] = struct{}{}
	}
	for _, mismatch := range args[2:] {
		if strings.TrimSpace(mismatch) == "" {
			return fmt.Errorf("empty boundary mismatch name")
		}
		if _, ok := mismatches[mismatch]; !ok {
			return fmt.Errorf("boundary report stage mismatch missing: %s/%s", stageName, mismatch)
		}
	}
	return nil
}

func readAdmissionLiveRouteBoundaryReportForAssert(path string) (admissionLiveRouteBoundaryReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteBoundaryReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("boundary report path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("boundary report not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("boundary report not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("boundary report JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("boundary report decode failed: %w", err)
	}
	return report, root, nil
}

func parseAdmissionLiveRouteBoundaryReportReceiptCount(value string) (int, error) {
	if value == "" {
		return 0, fmt.Errorf("expected receipt count must be numeric")
	}
	for _, ch := range value {
		if ch < '0' || ch > '9' {
			return 0, fmt.Errorf("expected receipt count must be numeric")
		}
	}
	return strconv.Atoi(value)
}

func admissionLiveRouteBoundaryReportStageByName(report admissionLiveRouteBoundaryReport, stageName string) (admissionLiveRouteBoundaryReportStage, int) {
	var found admissionLiveRouteBoundaryReportStage
	count := 0
	for _, stage := range report.Stages {
		if stage.Name == stageName {
			found = stage
			count++
		}
	}
	return found, count
}

func admissionLiveRouteBoundaryReportHasReason(report admissionLiveRouteBoundaryReport, reason string) bool {
	for _, got := range report.Reasons {
		if got == reason {
			return true
		}
	}
	return false
}

func admissionLiveRouteBoundaryReportHasObject(root map[string]json.RawMessage, key string) bool {
	raw, ok := root[key]
	if !ok {
		return false
	}
	return strings.HasPrefix(strings.TrimSpace(string(raw)), "{")
}

func admissionLiveRouteBoundaryReportSchemaError(report admissionLiveRouteBoundaryReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("boundary report schema missing")
	}
	if report.Schema != admissionLiveRouteBoundaryReportSchema {
		return fmt.Errorf("boundary report schema mismatch: got %q want %q", report.Schema, admissionLiveRouteBoundaryReportSchema)
	}
	return nil
}

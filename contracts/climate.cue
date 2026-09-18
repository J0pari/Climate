package climate

// Climate-owned scientific/research records. Cross-repository fields deliberately
// mirror the Commons semantic waist (run/trace/correlation/causation/build/
// fingerprint) without making Commons authoritative over climate semantics.

#Semver: =~"^[0-9]+\\.[0-9]+\\.[0-9]+([-+][0-9A-Za-z.-]+)?$"
#Sha256: =~"^sha256:[0-9a-f]{64}$"
#Id: =~"^[a-z][a-z0-9_.-]*$"
#Fingerprint: =~"^[0-9a-f]{64}$"

#Maturity: "concept" | "prototype" | "runnable" | "verified" | "validated" | "replicated" | "decision-eligible"
#EvidenceClass: "formal" | "observed" | "simulated" | "counterfactual" | "behavioral" | "heuristic"
#VerificationStatus: "unverified" | "verified" | "failed" | "inconclusive"
#EvidenceRelation: "supports" | "attacks" | "depends_on"
#DeterminismClass: "D0" | "D1" | "D2"
#NetworkPolicy: "none" | "restricted" | "required"
#ClaimType: "software" | "numerical" | "physical" | "statistical" | "predictive" | "causal" | "interpretive" | "performance" | "resource" | "interoperability"
#ConfigurationKind: "kernel_parameters" | "numerical_policy" | "data_policy" | "execution_policy"
#ConfigurationProvenance: "explicit" | "literature_fixed" | "calibrated" | "learned" | "experiment_policy"

#ConfigurationAuthorityRef: {
	source_id: #Id
	// Stable provider, publication, calibration-artifact, or model identity.
	identity: string & !=""
	digest?: #Sha256
}
#ExecutionClass: "R0_static" | "R1_portable_cpu" | "R2_toolchain_ci" | "R3_cuda_device" | "R4_integrated_system" | "R5_large_data"
#ExecutionFailureClass: "capability_unavailable" | "required_input_missing" | "configuration_invalid" | "implementation_unavailable" | "implementation_substituted" | "backend_mismatch" | "precision_mismatch" | "resource_mismatch" | "dataset_mismatch" | "undeclared_default"

#ResourceEnvelope: {
	cpu_cores?:      number & >0
	memory_bytes?:   int & >=0
	gpu_count?:      int & >=0
	vram_bytes?:     int & >=0
	compute_capability?: string
	network:         #NetworkPolicy
	max_seconds?:    int & >0
}

// Configuration records are deliberately owner-scoped. Shared immutable
// physical reference values are not runtime configuration and live in their
// focused scientific authorities instead of being copied into these records.
#ConfigurationRecord: {
	configuration_id: #Id
	semantic_version: #Semver
	kind:              #ConfigurationKind
	owner:             #Id
	provenance:        #ConfigurationProvenance
	authority_refs?:   [...#ConfigurationAuthorityRef]
	settings:          {...}

	if provenance == "literature_fixed" {
		authority_refs: [_, ...]
	}
	if provenance == "calibrated" {
		authority_refs: [_, ...]
	}
	if provenance == "learned" {
		authority_refs: [_, ...]
	}
}

#ConfigurationRef: {
	configuration_id: #Id
	semantic_version: #Semver
	kind:              #ConfigurationKind
	owner:             #Id
	provenance:        #ConfigurationProvenance
	record_path:       string & =~"^configurations/(kernel|numerical|data|execution)/[A-Za-z0-9._/-]+\\.json$"
	digest:            #Sha256
}

#KernelParameterRef: #ConfigurationRef & {
	kind: "kernel_parameters"
	record_path: =~"^configurations/kernel/"
}

#NumericalPolicyRef: #ConfigurationRef & {
	kind: "numerical_policy"
	record_path: =~"^configurations/numerical/"
}

#DataPolicyRef: #ConfigurationRef & {
	kind: "data_policy"
	record_path: =~"^configurations/data/"
}

#ExecutionPolicyRef: #ConfigurationRef & {
	kind: "execution_policy"
	record_path: =~"^configurations/execution/"
}

// The experiment/run surface composes references only. Component definitions
// remain in their owner-scoped records and are bound by content digest.
#ExperimentConfiguration: {
	kernel_parameters?:   [...#KernelParameterRef]
	numerical_policies?:  [...#NumericalPolicyRef]
	data_policies?:       [...#DataPolicyRef]
	execution_policies?:  [...#ExecutionPolicyRef]
}

#ExecutionIdentity: {
	method_id:            #Id
	implementation_id:    string & !=""
	implementation_build: string & !=""
	backend_id:           string & !=""
	precision:            string & !=""
	resource_class:       #ExecutionClass
}

#ExecutionResolution: {
	requested: #ExecutionIdentity
	status: "eligible" | "ineligible"

	if status == "eligible" {
		resolved: requested
		failure?: _|_
	}

	if status == "ineligible" {
		resolved?: _|_
		failure: #ExecutionFailureClass
	}
}

#AcceleratorContract: {
	implementation_id: string & !=""
	precision: {
		input:       string & !=""
		accumulator: string & !=""
		output:      string & !=""
		refinement?: string
	}
	determinism: #DeterminismClass
	rng?: {
		algorithm: string & !=""
		seed_policy: string & !=""
	}
	// Alternate implementations require distinct identities; this is not
	// permission for runtime substitution of an unavailable requested capability.
	alternate_identity_required: true
	resource: #ResourceEnvelope
}

#DatasetRef: {
	id:            #Id
	digest:        #Sha256
	source_family: string & !=""
	source_version?: string
	variables: [...string] & [_, ...]
	spatial_domain?: string
	grid?:           string
	vertical_coordinates?: string
	calendar?:       string
	time_domain?:    string
	temporal_resolution?: string
	units_metadata?: {...}
	quality_control_policy?: string
	preprocessing_fingerprint?: #Fingerprint
	license?: string
	citation?: string
	artifacts?: [...#ArtifactRef]
}

#MethodBuildIdentity: {
	policy: "source_digest_at_run"
	sources: [...string & !=""] & [_, ...]
}

#MethodDescriptor: {
	method_id:             #Id
	semantic_version:      #Semver
	build_identity?:       #MethodBuildIdentity
	contract_fingerprint?: #Fingerprint
	maturity:              #Maturity
	hypothesis_family:     string & !=""
	input_contracts:       [...string]
	output_contracts:      [...string]
	assumptions:           [...string]
	known_limitations:     [...string]
	resource:              #ResourceEnvelope
	accelerator?:          #AcceleratorContract
	reference_methods?:    [...#Id]
}

#MetricDefinition: {
	metric_id:        #Id
	semantic_version: #Semver
	units?:           string
	direction:        "higher" | "lower" | "target" | "descriptive"
	target?:          number
	aggregation_domain: string & !=""
	missing_data_policy: string & !=""
	uncertainty_method?: string

	if direction == "target" {
		target: number
	}
	if direction == "descriptive" {
		target?: _|_
	}
}

#MetricResultStatus: "finite" | "rank_deficient" | "undefined"

#MetricResult: {
	metric:  #MetricDefinition
	status:  #MetricResultStatus
	value?:  number
	detail?: string & !=""
	interval?: {
		lower: number
		upper: number
		level?: number & >0 & <1
	}
	sample_size?: int & >=0
	reference_population?: string

	if status == "finite" {
		value: number
	}
	if status != "finite" {
		value?: _|_
		interval?: _|_
		detail: string & !=""
	}
}

#ArtifactRef: {
	artifact_id: #Id
	digest:      #Sha256
	media_type:  string & !=""
	schema?:     string
	shape?:      [...int]
	bytes?:      int & >=0
	uri?:        string
}

#BudgetTermClass: "boundary_flux" | "resolved_physical_source" | "resolved_physical_sink" |
	"resolved_dissipation" | "internal_exchange" | "coupling_exchange" | "numerical_correction" |
	"solver_residual_effect" | "roundoff_estimate"

#BudgetIdentity: {
	budget_id:           string & !=""
	quantity_id:         string & !=""
	units:               string & !=""
	domain_id:           string & !=""
	interval_start_ns:   int & <interval_end_ns
	interval_end_ns:     int & >interval_start_ns
	precision:           string & !=""
	accumulation_method: string & !=""
}

#BudgetTermTotal: {
	class:                   #BudgetTermClass
	process_id:              string & !=""
	boundary_classification?: string & !=""
	signed_amount:           number
	absolute_uncertainty_bound?: number & >=0

	if class == "boundary_flux" {
		boundary_classification: string & !=""
	}
	if class == "resolved_physical_source" {
		signed_amount: number & >=0
	}
	if class == "resolved_physical_sink" {
		signed_amount: number & <=0
	}
	if class == "resolved_dissipation" {
		signed_amount: number & <=0
	}
}

#BudgetClassTotal: {
	class:         #BudgetTermClass
	signed_amount: number
}

#BudgetReport: {
	identity:    #BudgetIdentity
	partition_count: int & >=1
	quantity_before: number
	quantity_after:  number
	actual_change:   number
	accounted_change: number
	unexplained_residual: number
	conservative_absolute_uncertainty_bound?: number & >=0
	class_totals: [...#BudgetClassTotal]
	term_totals:  [...#BudgetTermTotal]
}

#MetricResultSet: {
	schema_version: 1
	metrics: [...#MetricResult] & [_, ...]
}

#ExperimentOutcome: {
	schema_version: 1
	experiment_id: #Id
	runs:      [...#RunManifest] & [_, ...]
	artifacts: [...#ArtifactRef] & [_, ...]
	metrics:   [...#MetricResult] & [_, ...]
	evidence:  [...#EvidenceRecord]
}

#ExperimentSpec: {
	experiment_id: #Id
	semantic_version: #Semver
	question: string & !=""
	hypothesis?: string
	candidate_methods: [...#Id] & [_, ...]
	baseline_methods:  [...#Id] & [_, ...]
	datasets:          [...#DatasetRef] & [_, ...]
	configuration?:    #ExperimentConfiguration
	primary_metrics:   [...#MetricDefinition] & [_, ...]
	secondary_metrics?: [...#MetricDefinition]
	negative_controls: [...string]
	falsifiers:        [...string]
	split_policy?:     string
	preprocessing?:    string
	seeds:             [...int]
	stopping_rule:     string & !=""
	multiple_comparison_policy?: string
	expected_artifact_types?: [...string]
	resource?: #ResourceEnvelope
}

#ResolvedEnvironment: {
	os?:      string
	arch?:    string
	container_digest?: #Sha256
	toolchains?: {...}
	libraries?: {...}
}

#HardwareRecord: {
	cpu?: string
	memory_bytes?: int & >=0
	gpus?: [...{
		model: string & !=""
		compute_capability?: string
		vram_bytes?: int & >=0
		driver?: string
	}]
}

#RunManifest: {
	run_id:              string & !=""
	trace_id?:           string
	correlation_id?:     string
	causation_id?:       string
	experiment_id:       #Id
	repository_revision: string & !=""
	producer_build:      string & !=""
	contract_fingerprint?: #Fingerprint
	method_builds:       [string]: string
	execution:           #ExecutionResolution
	scientific_output_eligible: bool
	resolved_dataset_digests: [...#Sha256]
	resolved_configuration: #ExperimentConfiguration
	seeds:               [...int]
	environment:         #ResolvedEnvironment
	hardware:            #HardwareRecord
	commands:            [...string] & [_, ...]
	exit_code:           int
	started_at?:         string
	ended_at?:           string
	stdout_digest?:      #Sha256
	stderr_digest?:      #Sha256
	artifacts:           [...#ArtifactRef]
	accelerator?: {
		implementation_id: string & !=""
		precision: string & !=""
		determinism: #DeterminismClass
		direct_execution: bool
	}

	if scientific_output_eligible {
		execution: {
			status: "eligible"
		}
		exit_code: 0
	}

	if execution.status == "ineligible" {
		scientific_output_eligible: false
		exit_code: int & !=0
	}
}

#EvidenceRecord: {
	evidence_id:      #Id
	claim_id:         #Id
	evidence_class:   #EvidenceClass
	verification:     #VerificationStatus
	relation:         #EvidenceRelation
	run_id?:          string
	artifacts:        [...#ArtifactRef]
	metrics:          [...#MetricResult]
	scope:            string & !=""
	threats_to_validity: [...string]
	notes?:           string
}

#Claim: {
	claim_id:          #Id
	statement:         string & !=""
	claim_type:        #ClaimType
	subject?:          string
	maturity:          #Maturity
	scope:             string & !=""
	depends_on:        [...#Id]
	required_evidence: [...string]
	supporting_evidence: [...#Id]
	attacking_evidence:  [...#Id]
	unresolved_falsifiers: [...string]
	owner?:            string
	supersedes?:       [...#Id]
	decision_policy?:  string
}

#CommonsSchedulerPin: {
	schema:      "gpu-scheduler/v1"
	fingerprint: #Fingerprint
	adoptedAt:   string & !=""
	capabilities: [...string & !=""]
	owner:       "commons"
}

#CommonsWorkSchedulerPin: {
	schema:      "work-scheduler/v1"
	fingerprint: #Fingerprint
	adoptedAt:   string & !=""
	capabilities: [...string & !=""]
	owner:       "commons"
}

#ClimateCommonsInterface: {
	schema_version: 1
	interface_id:   "climate.commons.v1"
	repository_id:  "J0pari/Climate"
	supported_control_level: "observe" | "read" | "write"
	scheduler_contract: {
		schema:   "work-scheduler/v1"
		pin_path: "contracts/work-scheduler-pin.json"
		owner:    "commons"
	}
	entrypoints: {
		inspect:           [...string & !=""] & [_, ...]
		run_experiment:    [...string & !=""] & [_, ...]
		submit_experiment: [...string & !=""] & [_, ...]
		live_read_witness: [...string & !=""] & [_, ...]
	}
	produces: [...string & !=""] & [_, ...]
	accepts:  [...string & !=""] & [_, ...]
	scientific_authority: {
		contract:    string & !=""
		experiments: string & !=""
		methods:     string & !=""
		claims:      string & !=""
		evidence:    string & !=""
	}
	limitations: [...string & !=""]
}

#CommonsEvaluationExchangePin: {
	schema:      "evaluation-exchange/v1"
	fingerprint: #Fingerprint
	adoptedAt:   string & !=""
	capabilities: [...string & !=""]
	owner:       "commons"
}


#ExternalModelRuntimeStatus: "unavailable" | "available"

#ClimateContractChoiceTask: {
	task_id: #Id
	prompt: string & !=""
	choices: {
		A: string & !=""
		B: string & !=""
		C: string & !=""
	}
	correct_choice: "A" | "B" | "C"
	authority: string & !=""
}

#ClimateContractReasoningTaskSet: {
	schema: "climate-contract-reasoning-taskset/v1"
	task_set_id: #Id
	semantic_version: #Semver
	scope: string & !=""
	tasks: [...#ClimateContractChoiceTask] & [_, ...]
}

#ExternalArtifactEvaluationSpec: {
	schema_version: 1
	evaluation_id: #Id
	semantic_version: #Semver
	evaluator_repository: "J0pari/Climate"
	subject_contract: "training.model-artifact/v1"
	prediction_contract: "climate-contract-reasoning-predictions/v1"
	result_contract: "climate-contract-reasoning-result/v1"
	adapter: {
		interface: "climate.external-model-runtime/v1"
		status: #ExternalModelRuntimeStatus
		required_capabilities: [...string & !=""] & [_, ...]
		implementation?: string & !=""
	}
	task_set: #ArtifactRef & {
		schema: "climate-contract-reasoning-taskset/v1"
		uri: =~"^fixtures/evaluation/[A-Za-z0-9._/-]+\\.json$"
	}
	primary_metrics: [...#MetricDefinition] & [_, ...]
	negative_controls: [...{
		control_id: #Id
		description: string & !=""
	}] & [_, ...]
	falsifiers: [...string & !=""] & [_, ...]
	evaluator_resource: #ResourceEnvelope
	subject_runtime_resource_policy: "must-be-declared-at-execution"
	evidence_policy: {
		evidence_class: "behavioral"
		verification: "verified"
		scope: string & !=""
	}
	issuance: {
		requires_adapter_available: true
		promotion_authority: "none"
	}
}


#ContractReasoningPrediction: {
	task_id: #Id
	choice: "A" | "B" | "C"
}

#ExternalContractReasoningPredictions: {
	schema: "climate-contract-reasoning-predictions/v1"
	evaluation_id: "external.training_artifact.climate_contract_reasoning.v1"
	subject_digest: #Fingerprint
	runtime: {
		interface: "climate.external-model-runtime/v1"
		implementation: string & !=""
		subject_digest: #Fingerprint
	}
	responses: [...#ContractReasoningPrediction] & [_, ...]
}

#ExternalContractReasoningResult: {
	schema: "climate-contract-reasoning-result/v1"
	evaluation_id: "external.training_artifact.climate_contract_reasoning.v1"
	subject_digest: #Fingerprint
	task_set_digest: #Sha256
	prediction_digest: #Sha256
	metrics: {
		contract_choice_accuracy: number & >=0 & <=1
		unsafe_semantic_upgrade_rate: number & >=0 & <=1
	}
	negative_controls: {
		permuted_answer_key: {
			accuracy: number & >=0 & <=1
			delta_vs_observed: number & >=-1 & <=1
		}
	}
	evidence: {
		evidence_class: "behavioral"
		verification: "verified"
		scope: string & !=""
	}
}

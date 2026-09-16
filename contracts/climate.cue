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
#ParameterProvenance: "placeholder" | "fallback" | "heuristic" | "calibrated" | "literature_fixed" | "learned"

#ResourceEnvelope: {
	cpu_cores?:      number & >0
	memory_bytes?:   int & >=0
	gpu_count?:      int & >=0
	vram_bytes?:     int & >=0
	compute_capability?: string
	network:         #NetworkPolicy
	max_seconds?:    int & >0
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
	fallback_identity_required: bool
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

#MethodDescriptor: {
	method_id:             #Id
	semantic_version:      #Semver
	implementation_build:  string & !=""
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
}

#MetricResult: {
	metric:  #MetricDefinition
	value:   number
	interval?: {
		lower: number
		upper: number
		level?: number & >0 & <1
	}
	sample_size?: int & >=0
	reference_population?: string
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

#ExperimentSpec: {
	experiment_id: #Id
	semantic_version: #Semver
	question: string & !=""
	hypothesis?: string
	candidate_methods: [...#Id] & [_, ...]
	baseline_methods:  [...#Id] & [_, ...]
	datasets:          [...#DatasetRef] & [_, ...]
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
	resolved_dataset_digests: [...#Sha256]
	resolved_configuration: {...}
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

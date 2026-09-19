# Sheaf / cohomology realization contract

Status: architectural and mathematical specification for `climate.sheaf.consistency`.

## Mathematical contract

Sheaf/cohomology terminology is reserved for structures satisfying the defining laws. Mathematical realization, scalable execution, and empirical climate usefulness are distinct authorities.

1. **Declared base complex.** Climate cochains live on an explicit simplicial/cellular complex or poset.
2. **Linear restrictions and cochains.** Stalks, units, missingness, and restriction maps are explicit; coboundaries are assembled from those restrictions.
3. **Functorial restrictions.** Identity and composition laws are executable invariants where higher cells are present.
4. **Cochain-complex identity.** Consecutive production coboundaries satisfy `d_(k+1) o d_k = 0`.
5. **Cohomology-derived invariants.** Betti/cohomology terminology refers to kernel/image structure over a declared coefficient system, not thresholded residual counts.
6. **Gluing semantics.** Compatible local sections and global-section/nullspace structure are defined by restrictions. Reconstruction or interpolation is separate.
7. **Scale-preserving realization.** Climate-data execution must not require a global all-pairs scan, one projected CRS, global tuple/powerset enumeration, dense global coboundary matrices, or loading all stations and times into memory.
8. **Reference/production separation.** Small exact implementations are independent correctness oracles. They do not discharge production climate-data obligations.

## Canonical station substrate

`src/station_sheaf.py` is provider-neutral. It consumes stable station identities, WGS84 coordinates, explicit variable/unit descriptors, and bounded observation sections with a separate observed/missing mask.

Global station locality uses a 3-D unit-sphere `cKDTree` for conservative radius candidates and `pyproj.Geod` for exact WGS84 ellipsoidal refinement. Anti-meridian and polar behavior therefore do not depend on a local projected CRS. Edge ownership is deterministic over half-open tail-vertex ranges so shards own disjoint edges over the same immutable catalog.

The current base complex is a declared Vietoris--Rips complex over that geodesic graph. Higher simplices are generated only when requested from intersections of forward local adjacency. Simplices are owned by their smallest vertex, giving deterministic partitioning; a partition can request the codimension-one halo faces needed by its owned cofaces.

`StationSchemaSheaf` supports heterogeneous station capabilities. A simplex stalk is the intersection of the normalized variable schemas carried by its vertices. Restrictions are sparse coordinate-selection maps, so nontrivial heterogeneous restrictions retain exact identity/composition semantics. Cochain indexes are partition-local, and sparse oriented coboundaries are assembled from owned cofaces plus explicit halo faces.

`StationIdentitySheaf` is the same-variable special case. Degree-zero coboundaries are SciPy CSR with two structural nonzeros per edge-variable row, or restrictions are applied locally without materializing a global matrix. Missing observations mask restriction rows; they are never converted into observations with a zero/default value.

## Generic canonical sheaf operator

`src/sheaf.rs` owns generic real finite-dimensional vertex/edge stalks with explicit non-identity restriction matrices. Ordinary residual and `L0` application are matrix-free over local edge blocks. Sparse coboundary triplets are available for maintained sparse-library backends.

Dense `D`, `D^T D`, and SVD materialization remain bounded verification diagnostics. They are not the production foundation for large climate-data graphs.

## Global sections and gluing

For the coordinate-selection station sheaf, each normalized variable induces an identity sheaf on the subgraph of stations that structurally support it. Global-section dimension is therefore computed from sparse connected components of those induced subgraphs rather than a dense global nullspace.

A partial observed station section extends to a global section exactly when all observed values agree inside each relevant connected component. Components with an observation anchor are determined; unanchored components remain explicitly free; conflicting components make extension fail. No interpolation or default value is inserted and called gluing.

## Provider composition

`reference/station_temperature_sheaf.py` parses the captured NCEI GHCN-Daily witness and projects provider identity, coordinates, TMAX/TMIN, units, quality flags, and missingness into the canonical `StationCatalog` and `StationSection`.

The three-station artifact is only a provider/provenance regression fixture. It owns no second topology builder, projected coordinate system, dense cochain implementation, or station-scale execution model.

The data architecture targets a federation of every station worldwide that can be lawfully and reproducibly accessed without paid data access. Provider aliases, relocation/history, revisions, quality/source flags, access/license semantics, and cross-provider deduplication remain explicit data semantics. Numerical kernels do not encode a fixed provider list.

`architecture/station_providers.json` is the provider-discovery authority and explicitly prohibits a hand-selected station list. Each registered provider also repeats the resolved data authority's access, license/access-constraint, and update semantics under machine-enforced equality and declares its federation-specific revision-identity rule, so provider discovery cannot silently weaken upstream provenance. `src/station_federation.py` owns the provider-neutral identity/storage seam: one immutable root alias gives a stable canonical station id; later aliases require evidence-bound crosswalks and do not mutate that id; crosswalk application resolves the declared root alias, rejects alias ownership conflicts, records the crosswalk artifact digest on the alias binding, and can reverse exactly those bindings by evidence digest; resolved location epochs preserve their source artifact; global manifests reference bounded content-addressed catalog shards and provider/time/variable/spatial observation partitions rather than embedding the station-by-time tensor. Provider revisions append explicit supersession or tombstone lineage instead of overwriting prior observations, and coverage summaries are derived from active manifest references so gaps remain observable.

`data/ncei_ghcnd_bulk.py` is the first federation-scale observation provider parser/router. `data/ncei_ghcnh_bulk.py` adds the second provider catalog: it consumes externally captured NCEI station-list and annual archive artifacts, validates the provider's version/data-year/creation-date filename identity, parses the documented global station list as a distinct namespace, uses exact shared GHCN identifiers as digest-bound crosswalk evidence, and creates separate roots for unmatched hourly stations. Captured annual tar archives are streamed station-member by station-member and published as immutable spatial/year partitions whose source revision is the compressed archive SHA-256. Every PSV provider field remains unchanged; acquisition, transport durability, unit conversion, code interpretation, and quality-policy filtering are separate external/downstream responsibilities.

`data/ncei_ghcnd_bulk.py` is the first federation-scale provider parser/router. It begins only after station, inventory, or by-year artifacts have been captured by an external data-transfer capability. Climate preserves provider URL/version identity and imported artifact digests, parses the complete station/inventory formats, maps capability metadata into federation identity, builds bounded adaptive catalog shards, and streams captured by-year rows directly to caller-owned spatial/year/element partition sinks. Network transfer, byte-range resume, remote-validator handling, retry policy, and download checkpointing are deliberately outside Climate. The subset REST reference likewise parses captured bytes and reconstructs request identity but performs no live retrieval. The row path validates provider year/date contracts, preserves missing values and measurement/quality/source flags, and resolves every provider station through the captured federation catalog without year-sized materialization.

`data/ncei_ghcnd_parquet.py` is the durable GHCN publication adapter behind that sink boundary. It buffers only a bounded cross-partition row batch, writes Parquet fragments with the pinned Apache Arrow implementation, binds every partition to the SHA-256 identity of the captured compressed by-year artifact, and atomically publishes a content-addressed partition object whose manifest records fragment digests, raw-value semantics, row/time bounds, writer identity, and explicit supersession lineage. Complete flushed fragment batches are resumable: an atomic checkpoint records fragment metadata plus fsynced logical/input-prefix sidecars, and a resumed run must replay the identical canonical station/shard-resolved prefix before appending. Replaying identical input is idempotent; a changed provider revision creates a distinct immutable object and must name the predecessor when both revisions enter one federation manifest. Unit normalization remains outside this provider-native storage boundary.

The sheaf path is not intended to become another generic station-QC, interpolation, metadata-normalization, or provider-ingestion framework. Provider-native quality flags and upstream decoding remain authoritative inputs; existing QC/graph residual/kriging/interpolation methods are explicit baselines rather than functionality to absorb into the sheaf implementation. An uncertainty-aware or cohomological quantity earns a scientific role only if controlled faults and real-data experiments show information beyond those simpler/native diagnostics, especially higher-order incompatibility or localization that pairwise residuals miss.

When station or gridded observations arrive through an external evaluation/preprocessing system with adequate provenance, Climate should preserve that native transformation chain and construct the sheaf over the resulting declared observation objects rather than replay the same generic preprocessing under a second implementation.
## Independent mathematical references

`reference/sheaf_cohomology.py` retains exact finite-complex, finite-cover nerve, GF(2) cellular-sheaf, cohomology-dimension, and functoriality witnesses. Its powerset enumeration and dense exact matrices are useful because the fixtures are deliberately bounded and independently inspectable; they are not a station-network scaling path.

The scalar identity-restriction special case remains kernel-checked in Lean. That proof strengthens the algebraic boundary without changing production implementation identity.

## Current realization boundary

The canonical path realizes global indexed station locality, deterministic partition ownership, bounded observation sections, sparse degree-zero operators, heterogeneous coordinate-selection restrictions, partition-local sparse higher-cochain assembly, matrix-free generic degree-zero application, and exact component-local global-section extension semantics.

The machine realization ledger remains authoritative about which of those capabilities have executable promotion witnesses at the current revision. Empirical incremental value over ordinary QC, graph residuals, interpolation/kriging, and other declared baselines remains a separate claim.

Worldwide station scale is an engineering/data target, not empirical validation.

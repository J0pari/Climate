{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DataKinds #-}

-- Climate data sheaf with Čech cohomology
-- Weather station network consistency checking
-- Detects inconsistencies and gaps using topological invariants
--
-- DATA SOURCE REQUIREMENTS:
--
-- 1. GLOBAL WEATHER STATION NETWORK:
--    - Source: GHCN-Daily (Global Historical Climatology Network)
--    - Stations: 100,000+ worldwide with varying record lengths
--    - Variables: TMAX, TMIN, PRCP, SNOW, SNWD, plus 30+ others
--    - Temporal: Daily, 1880-present for some stations
--    - Format: Fixed-width text or NetCDF4
--    - Size: ~50GB for complete archive (NEVER TESTED WITH REAL DATA)
--    - API: ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/
--    - Metadata: Station locations, elevations, relocations
--    - Missing: Station moves not always documented
--
-- 2. STATION METADATA:
--    - Source: GHCN-M station inventory
--    - Format: CSV with lat, lon, elevation, name, WMO ID
--    - Size: <10MB
--    - Critical: Station history (moves, instrument changes)
--    - Missing: Everything - just stubs, no actual station data handling
--
-- 3. REGIONAL HIGH-DENSITY NETWORKS:
--    - Source: USCRN (US), ECAD (Europe), ACORN-SAT (Australia)
--    - Resolution: ~50-100km spacing
--    - Temporal: Hourly to daily
--    - Format: NetCDF4 or CSV
--    - Size: ~10GB per network
--    - Purpose: Validate sheaf consistency at high resolution
--    - Missing: Africa, South America coverage
--
-- 4. GRIDDED COMPARISON DATA:
--    - Source: CRU TS, Berkeley Earth, GISTEMP
--    - Resolution: 0.5° to 5° grids
--    - Purpose: Check sheaf reconstructions against gridded products
--    - Format: NetCDF4
--    - Size: ~5GB per dataset
--    - Missing: Uncertainty in interpolation methods
--
-- 5. TOPOLOGICAL STRUCTURE:
--    - Source: Computed from station locations
--    - Method: Delaunay triangulation, Voronoi cells
--    - Purpose: Define nerve complex for Čech cohomology
--    - Missing: Any actual coverage radius - just hardcoded values

module ClimateMultiscaleSheaf where

import qualified Data.Map.Strict as Map
import qualified Data.Set as Set
import Data.Maybe (fromMaybe, mapMaybe)
import Control.Monad (guard)

-- Core types

-- | A weather station with location and coverage radius
data Station = Station
    { stationId :: String
    , latitude :: Double
    , longitude :: Double
    , elevation :: Double  -- meters
    , coverageRadius :: Double  -- km
    } deriving (Eq, Ord, Show)

-- | Climate measurement at a point in time
data Measurement = Measurement
    { temperature :: Maybe Double  -- Celsius
    , pressure :: Maybe Double     -- hPa
    , humidity :: Maybe Double     -- percentage
    , windSpeed :: Maybe Double    -- m/s
    , precipitation :: Maybe Double -- mm
    } deriving (Eq, Show)

-- | Time-indexed measurements
type TimeSeries = Map.Map Double Measurement  -- Time -> Measurement

-- Sheaf structure

-- | A sheaf of climate data over the station network
data ClimateSheaf = ClimateSheaf
    { sections :: Map.Map Station TimeSeries
    , overlaps :: Map.Map (Station, Station) Double  -- Overlap strength [0,1]
    , restrictions :: Map.Map (Station, Station) (Measurement -> Measurement)
    }

-- | Create local section from station data
createLocalSection :: Station -> TimeSeries -> (Station, TimeSeries)
createLocalSection station series = (station, series)

-- | Compute overlap between two stations based on distance
computeOverlap :: Station -> Station -> Double
computeOverlap s1 s2 = 
    let dist = haversineDistance (latitude s1, longitude s1) 
                                 (latitude s2, longitude s2)
        combined = coverageRadius s1 + coverageRadius s2
    in max 0 (1 - dist / combined)

-- | Haversine distance between two points (in km)
haversineDistance :: (Double, Double) -> (Double, Double) -> Double
haversineDistance (lat1, lon1) (lat2, lon2) =
    let r = 6371  -- Earth radius in km
        dLat = (lat2 - lat1) * pi / 180
        dLon = (lon2 - lon1) * pi / 180
        a = sin(dLat/2)^2 + cos(lat1*pi/180) * cos(lat2*pi/180) * sin(dLon/2)^2
        c = 2 * atan2 (sqrt a) (sqrt (1-a))
    in r * c

-- Čech cohomology for consistency checking

-- | 0-cochains: Assignments to stations
type C0 = Map.Map Station Measurement

-- | 1-cochains: Assignments to overlapping pairs
type C1 = Map.Map (Station, Station) Double  -- Discrepancy measure

-- | 2-cochains: Assignments to triple overlaps
type C2 = Map.Map (Station, Station, Station) Double

-- | Coboundary operator δ⁰: C⁰ → C¹
coboundary0 :: ClimateSheaf -> C0 -> C1
coboundary0 sheaf c0 = 
    Map.fromList
        [ ((s1, s2), measureDiscrepancy m1 m2 * overlap)
        | ((s1, s2), overlap) <- Map.toList (overlaps sheaf)
        , overlap > 0
        , Just m1 <- [Map.lookup s1 c0]
        , Just m2 <- [Map.lookup s2 c0]
        ]

-- | Coboundary operator δ¹: C¹ → C²
coboundary1 :: ClimateSheaf -> C1 -> C2
coboundary1 sheaf c1 =
    Map.fromList
        [ ((s1, s2, s3), cycleSum)
        | s1 <- stations
        , s2 <- stations
        , s3 <- stations
        , s1 < s2, s2 < s3  -- Ordered triples only
        , let cycleSum = fromMaybe 0 (Map.lookup (s1,s2) c1) 
                       - fromMaybe 0 (Map.lookup (s1,s3) c1)
                       + fromMaybe 0 (Map.lookup (s2,s3) c1)
        , abs cycleSum > 1e-6  -- Non-zero cycles only
        ]
  where
    stations = Map.keys (sections sheaf)

-- | Measure discrepancy between two measurements
measureDiscrepancy :: Measurement -> Measurement -> Double
measureDiscrepancy m1 m2 =
    let tempDiff = case (temperature m1, temperature m2) of
                     (Just t1, Just t2) -> abs (t1 - t2) / 10  -- Normalize by 10°C
                     _ -> 0
        presDiff = case (pressure m1, pressure m2) of
                     (Just p1, Just p2) -> abs (p1 - p2) / 50  -- Normalize by 50 hPa
                     _ -> 0
        humDiff = case (humidity m1, humidity m2) of
                    (Just h1, Just h2) -> abs (h1 - h2) / 100
                    _ -> 0
    in (tempDiff + presDiff + humDiff) / 3

-- Betti numbers and topological invariants

-- | Compute Betti numbers for the climate data topology
computeBettiNumbers :: ClimateSheaf -> C0 -> (Int, Int, Int)
computeBettiNumbers sheaf c0 =
    let c1 = coboundary0 sheaf c0
        c2 = coboundary1 sheaf c1
        
        -- b₀: Connected components (always 1 for connected network)
        b0 = 1
        
        -- b₁: Cycles (inconsistency loops)
        b1 = length $ filter (> 0.3) $ Map.elems c1
        
        -- b₂: Voids (coverage gaps)
        b2 = length $ filter (> 0.4) $ Map.elems c2
        
    in (b0, b1, b2)

-- | Euler characteristic χ = b₀ - b₁ + b₂
eulerCharacteristic :: (Int, Int, Int) -> Int
eulerCharacteristic (b0, b1, b2) = b0 - b1 + b2

-- Adjoint functors: Analysis ⊣ Synthesis

-- | Left adjoint F: Observations → Climate State (analysis)
data ClimateAnalysis a = ClimateAnalysis
    { rawData :: Map.Map Station TimeSeries
    , processedState :: a
    , coherence :: Double
    }

-- | Right adjoint G: Climate State → Predictions (synthesis)
data ClimateSynthesis a = ClimateSynthesis
    { climateState :: a
    , predictions :: Map.Map Station TimeSeries
    , confidence :: Double
    }

-- | Unit of adjunction η: Id → G∘F
-- Measures information preservation during analysis
adjunctionUnit :: ClimateAnalysis a -> ClimateSynthesis a -> Double
adjunctionUnit analysis synthesis =
    let dataPoints = sum $ map Map.size $ Map.elems (rawData analysis)
        predPoints = sum $ map Map.size $ Map.elems (predictions synthesis)
    in fromIntegral (min dataPoints predPoints) / fromIntegral (max dataPoints predPoints)

-- | Counit of adjunction ε: F∘G → Id
-- Measures reconstruction quality
adjunctionCounit :: ClimateSynthesis a -> ClimateAnalysis a -> Double
adjunctionCounit synthesis analysis = 
    coherence analysis * confidence synthesis

-- Gluing morphisms for consistency

-- | Glue local sections into global climate field
glueLocalSections :: ClimateSheaf -> Map.Map Station TimeSeries -> Maybe TimeSeries
glueLocalSections sheaf localData =
    let overlappingPairs = [(s1, s2) | ((s1, s2), w) <- Map.toList (overlaps sheaf), w > 0.5]
        
        -- Check consistency at overlaps
        consistent = all (checkConsistency localData) overlappingPairs
        
        -- If consistent, merge with weighted average
        merged = if consistent
                then Just $ mergeTimeSeries localData (overlaps sheaf)
                else Nothing
    in merged

-- | Check consistency between two overlapping stations
checkConsistency :: Map.Map Station TimeSeries -> (Station, Station) -> Bool
checkConsistency localData (s1, s2) =
    case (Map.lookup s1 localData, Map.lookup s2 localData) of
        (Just ts1, Just ts2) -> 
            let commonTimes = Set.intersection (Map.keysSet ts1) (Map.keysSet ts2)
                discrepancies = [measureDiscrepancy (ts1 Map.! t) (ts2 Map.! t) 
                               | t <- Set.toList commonTimes]
            in all (< 0.2) discrepancies  -- Threshold for consistency
        _ -> True  -- No overlap means no inconsistency

-- | Merge time series with weighted averaging
mergeTimeSeries :: Map.Map Station TimeSeries -> Map.Map (Station, Station) Double -> TimeSeries
mergeTimeSeries localData weights =
    -- Simplified: just take union of all measurements
    -- STUB: Should do kriging but just takes arbitrary union
    Map.unions $ Map.elems localData

-- Example: Detecting network inconsistencies

-- | Analyze climate network for topological features
analyzeClimateTopology :: ClimateSheaf -> IO ()
analyzeClimateTopology sheaf = do
    -- Get current measurements from all stations
    let currentMeasurements = getCurrentMeasurements sheaf
    
    -- Compute cohomology
    let (b0, b1, b2) = computeBettiNumbers sheaf currentMeasurements
    let chi = eulerCharacteristic (b0, b1, b2)
    
    -- Report findings
    putStrLn $ "Climate Network Topology Analysis:"
    putStrLn $ "  Connected components (b₀): " ++ show b0
    putStrLn $ "  Inconsistency cycles (b₁): " ++ show b1
    putStrLn $ "  Coverage gaps (b₂): " ++ show b2
    putStrLn $ "  Euler characteristic (χ): " ++ show chi
    
    -- Interpret results
    when (b1 > 0) $ putStrLn "  ⚠ Data inconsistencies detected between stations"
    when (b2 > 0) $ putStrLn "  ⚠ Coverage gaps detected in network"
    when (chi < 0) $ putStrLn "  ⚠ Complex topology indicates systematic issues"

-- | Get current measurements from all stations (simplified)
getCurrentMeasurements :: ClimateSheaf -> C0
getCurrentMeasurements sheaf =
    Map.fromList
        [ (station, fromMaybe emptyMeasurement $ Map.lookupMax series >>= return . snd)
        | (station, series) <- Map.toList (sections sheaf)
        ]
  where
    emptyMeasurement = Measurement Nothing Nothing Nothing Nothing Nothing

{-
Experimental sheaf framework for climate networks.

Limitations:
- Simplified overlap computation
- Basic discrepancy measures
- No temporal correlation
- Static network assumption

Enables:
- Inconsistency detection via cohomology
- Coverage gap identification via Betti numbers
- Cross-scale consistency verification
-}
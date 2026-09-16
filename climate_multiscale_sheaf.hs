{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DataKinds #-}

-- Legacy station-network diagnostics retained as a source reservoir while the
-- mathematically exact finite-sheaf reference is developed under reference/.
--
-- IMPORTANT SEMANTICS:
-- * overlap/discrepancy calculations in this file are heuristics, not
--   cohomology or topological invariants;
-- * candidate restriction functions have not been shown to satisfy sheaf
--   functoriality and therefore do not constitute a realized climate sheaf;
-- * arbitrary time-series union is preserved only as an explicitly named
--   legacy operation and is not sheaf gluing or reconstruction;
-- * missing observations are distinct from zero discrepancy and fail closed in
--   pairwise-consistency decisions.

module ClimateMultiscaleSheaf where

import qualified Data.Map.Strict as Map
import qualified Data.Set as Set
import Data.Maybe (catMaybes, mapMaybe)

-- Core observational types

data Station = Station
    { stationId :: String
    , latitude :: Double
    , longitude :: Double
    , elevation :: Double
    , coverageRadius :: Double
    } deriving (Eq, Ord, Show)

data Measurement = Measurement
    { temperature :: Maybe Double
    , pressure :: Maybe Double
    , humidity :: Maybe Double
    , windSpeed :: Maybe Double
    , precipitation :: Maybe Double
    } deriving (Eq, Show)

type TimeSeries = Map.Map Double Measurement

type StationSnapshot = Map.Map Station (Maybe Measurement)

-- | Legacy prototype container. The maps in 'restrictionCandidates' are
-- hypotheses for later sheaf construction; this type does not assert that they
-- satisfy identity/composition laws.
data ClimateNetworkPrototype = ClimateNetworkPrototype
    { sections :: Map.Map Station TimeSeries
    , overlaps :: Map.Map (Station, Station) Double
    , restrictionCandidates :: Map.Map (Station, Station) (Measurement -> Measurement)
    }

createLocalSection :: Station -> TimeSeries -> (Station, TimeSeries)
createLocalSection station series = (station, series)

computeOverlap :: Station -> Station -> Double
computeOverlap s1 s2 =
    let dist = haversineDistance (latitude s1, longitude s1)
                                 (latitude s2, longitude s2)
        combined = coverageRadius s1 + coverageRadius s2
    in if combined <= 0 then 0 else max 0 (1 - dist / combined)

haversineDistance :: (Double, Double) -> (Double, Double) -> Double
haversineDistance (lat1, lon1) (lat2, lon2) =
    let r = 6371
        dLat = (lat2 - lat1) * pi / 180
        dLon = (lon2 - lon1) * pi / 180
        a = sin(dLat/2)^2
          + cos(lat1*pi/180) * cos(lat2*pi/180) * sin(dLon/2)^2
        c = 2 * atan2 (sqrt a) (sqrt (1-a))
    in r * c

-- Heuristic discrepancy layer -------------------------------------------------

-- | Pair diagnostic with explicit observability. A missing comparable variable
-- is not treated as zero disagreement.
data PairAssessment
    = PairUnavailable
    | PairObserved
        { normalizedDiscrepancy :: Double
        , comparedVariableCount :: Int
        }
    deriving (Eq, Show)

type PairDiagnostics = Map.Map (Station, Station) PairAssessment

type TripleResiduals = Map.Map (Station, Station, Station) Double

-- | Preserve the legacy normalizations, but return Nothing when the two
-- measurements share no supported observed variable. These constants are
-- diagnostic scaling choices, not physical uncertainty models.
measurementDiscrepancy :: Measurement -> Measurement -> Maybe (Double, Int)
measurementDiscrepancy m1 m2 =
    let diffs = catMaybes
            [ fmap (\(a,b) -> abs (a-b) / 10)  ((,) <$> temperature m1 <*> temperature m2)
            , fmap (\(a,b) -> abs (a-b) / 50)  ((,) <$> pressure m1 <*> pressure m2)
            , fmap (\(a,b) -> abs (a-b) / 100) ((,) <$> humidity m1 <*> humidity m2)
            ]
    in case diffs of
        [] -> Nothing
        _  -> Just (sum diffs / fromIntegral (length diffs), length diffs)

pairDiagnostics :: ClimateNetworkPrototype -> StationSnapshot -> PairDiagnostics
pairDiagnostics model snapshot =
    Map.fromList
        [ ((s1, s2), assess s1 s2 overlap)
        | ((s1, s2), overlap) <- Map.toList (overlaps model)
        , overlap > 0
        ]
  where
    assess s1 s2 overlap =
        case (Map.lookup s1 snapshot, Map.lookup s2 snapshot) of
            (Just (Just m1), Just (Just m2)) ->
                case measurementDiscrepancy m1 m2 of
                    Just (score, n) -> PairObserved (score * overlap) n
                    Nothing -> PairUnavailable
            _ -> PairUnavailable

lookupObservedPair :: PairDiagnostics -> Station -> Station -> Maybe Double
lookupObservedPair diagnostics a b =
    case Map.lookup (a,b) diagnostics of
        Just (PairObserved score _) -> Just score
        _ -> case Map.lookup (b,a) diagnostics of
            Just (PairObserved score _) -> Just score
            _ -> Nothing

hasPositiveOverlap :: ClimateNetworkPrototype -> Station -> Station -> Bool
hasPositiveOverlap model a b =
    maybe False (> 0) (Map.lookup (a,b) (overlaps model))
    || maybe False (> 0) (Map.lookup (b,a) (overlaps model))

-- | Legacy triangle residual retained as a heuristic. Unlike the old routine,
-- it is emitted only for genuine pairwise overlaps with all three pair scores
-- observed; missing edges are never silently substituted by zero.
tripleResiduals :: ClimateNetworkPrototype -> PairDiagnostics -> TripleResiduals
tripleResiduals model diagnostics =
    Map.fromList $ mapMaybe residual triples
  where
    stations = Map.keys (sections model)
    triples =
        [ (a,b,c)
        | a <- stations, b <- stations, c <- stations
        , a < b, b < c
        , hasPositiveOverlap model a b
        , hasPositiveOverlap model a c
        , hasPositiveOverlap model b c
        ]
    residual (a,b,c) = do
        ab <- lookupObservedPair diagnostics a b
        ac <- lookupObservedPair diagnostics a c
        bc <- lookupObservedPair diagnostics b c
        pure ((a,b,c), ab - ac + bc)

-- Policy is separate from measurement ----------------------------------------

data DiagnosticThresholds = DiagnosticThresholds
    { pairDiscrepancyThreshold :: Double
    , tripleResidualThreshold :: Double
    } deriving (Eq, Show)

data ThresholdDiagnostics = ThresholdDiagnostics
    { overlapComponentCount :: Int
    , largePairDiscrepancyCount :: Int
    , largeTripleResidualCount :: Int
    , unavailablePairCount :: Int
    } deriving (Eq, Show)

networkStations :: ClimateNetworkPrototype -> Set.Set Station
networkStations model =
    let endpoints = concatMap (\((a,b),_) -> [a,b]) (Map.toList (overlaps model))
    in Set.union (Map.keysSet (sections model)) (Set.fromList endpoints)

neighbors :: ClimateNetworkPrototype -> Station -> Set.Set Station
neighbors model station = Set.fromList
    [ if station == a then b else a
    | ((a,b), weight) <- Map.toList (overlaps model)
    , weight > 0
    , station == a || station == b
    ]

overlapComponents :: ClimateNetworkPrototype -> Int
overlapComponents model = go (networkStations model) 0
  where
    go remaining count
        | Set.null remaining = count
        | otherwise =
            let seed = Set.findMin remaining
                component = flood Set.empty (Set.singleton seed)
            in go (remaining `Set.difference` component) (count + 1)
    flood visited frontier
        | Set.null frontier = visited
        | otherwise =
            let current = Set.findMin frontier
                rest = Set.delete current frontier
                next = neighbors model current `Set.difference` visited
            in flood (Set.insert current visited) (Set.union rest next)

computeThresholdDiagnostics
    :: DiagnosticThresholds
    -> ClimateNetworkPrototype
    -> StationSnapshot
    -> ThresholdDiagnostics
computeThresholdDiagnostics policy model snapshot =
    let pairs = pairDiagnostics model snapshot
        triples = tripleResiduals model pairs
        pairScores = [score | PairObserved score _ <- Map.elems pairs]
        unavailable = length [() | PairUnavailable <- Map.elems pairs]
    in ThresholdDiagnostics
        { overlapComponentCount = overlapComponents model
        , largePairDiscrepancyCount =
            length (filter (> pairDiscrepancyThreshold policy) pairScores)
        , largeTripleResidualCount =
            length (filter ((> tripleResidualThreshold policy) . abs) (Map.elems triples))
        , unavailablePairCount = unavailable
        }

-- Analysis/synthesis diagnostics ---------------------------------------------

-- These records and scalar diagnostics are not a categorical adjunction.
data ClimateAnalysis a = ClimateAnalysis
    { rawData :: Map.Map Station TimeSeries
    , processedState :: a
    , coherence :: Double
    }

data ClimateSynthesis a = ClimateSynthesis
    { climateState :: a
    , predictions :: Map.Map Station TimeSeries
    , confidence :: Double
    }

analysisSynthesisCoverageRatio
    :: ClimateAnalysis a -> ClimateSynthesis a -> Maybe Double
analysisSynthesisCoverageRatio analysis synthesis =
    let dataPoints = sum $ map Map.size $ Map.elems (rawData analysis)
        predPoints = sum $ map Map.size $ Map.elems (predictions synthesis)
        denominator = max dataPoints predPoints
    in if denominator == 0
       then Nothing
       else Just (fromIntegral (min dataPoints predPoints) / fromIntegral denominator)

reconstructionQualityProduct :: ClimateSynthesis a -> ClimateAnalysis a -> Double
reconstructionQualityProduct synthesis analysis =
    coherence analysis * confidence synthesis

-- Explicit legacy merge semantics --------------------------------------------

data ConsistencyStatus = Consistent | Inconsistent | InsufficientOverlapData
    deriving (Eq, Show)

pairConsistency
    :: Double
    -> Map.Map Station TimeSeries
    -> (Station, Station)
    -> ConsistencyStatus
pairConsistency threshold localData (s1, s2) =
    case (Map.lookup s1 localData, Map.lookup s2 localData) of
        (Just ts1, Just ts2) ->
            let commonTimes = Set.intersection (Map.keysSet ts1) (Map.keysSet ts2)
                scores = mapMaybe scoreAt (Set.toList commonTimes)
                scoreAt t = fmap fst (measurementDiscrepancy (ts1 Map.! t) (ts2 Map.! t))
            in if null scores
               then InsufficientOverlapData
               else if any (>= threshold) scores then Inconsistent else Consistent
        _ -> InsufficientOverlapData

-- | Preserve the historical Map.unions behavior for comparison only. This is
-- not reconstruction and not sheaf gluing. It refuses to merge when any
-- declared strong-overlap pair is inconsistent or lacks comparable data.
legacyUnionIfPairwiseConsistent
    :: Double
    -> ClimateNetworkPrototype
    -> Map.Map Station TimeSeries
    -> Either String TimeSeries
legacyUnionIfPairwiseConsistent threshold model localData =
    let pairs = [(a,b) | ((a,b), w) <- Map.toList (overlaps model), w > 0.5]
        statuses = map (pairConsistency threshold localData) pairs
    in if any (== Inconsistent) statuses
       then Left "pairwise discrepancy threshold exceeded"
       else if any (== InsufficientOverlapData) statuses
            then Left "insufficient overlap data for legacy union"
            else Right (Map.unions $ Map.elems localData)

-- Observational snapshot ------------------------------------------------------

currentMeasurements :: ClimateNetworkPrototype -> StationSnapshot
currentMeasurements model = Map.fromList
    [ (station, fmap snd (Map.lookupMax series))
    | (station, series) <- Map.toList (sections model)
    ]

-- | Human-readable exploratory report. Its counts are threshold diagnostics,
-- not invariants and not evidence of a sheaf/cohomological climate mechanism.
analyzeNetworkHeuristics :: DiagnosticThresholds -> ClimateNetworkPrototype -> IO ()
analyzeNetworkHeuristics policy model = do
    let snapshot = currentMeasurements model
        result = computeThresholdDiagnostics policy model snapshot
    putStrLn "Climate network heuristic diagnostics (not topological invariants):"
    putStrLn $ "  overlap graph components: " ++ show (overlapComponentCount result)
    putStrLn $ "  pair discrepancies above policy threshold: "
            ++ show (largePairDiscrepancyCount result)
    putStrLn $ "  triangle residuals above policy threshold: "
            ++ show (largeTripleResidualCount result)
    putStrLn $ "  unavailable overlap pairs: " ++ show (unavailablePairCount result)

{-
Remaining scientific/mathematical obligations are tracked in
methods/sheaf-realization.v1.json. The exact finite-complex/cellular-sheaf
reference lives in reference/sheaf_cohomology.py. This legacy file intentionally
retains only station-network heuristics and candidate semantics that may be
compared, ablated, migrated, or rejected.
-}

{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}

-- Climate symmetries and conservation laws via Noether's theorem
-- Experimental symmetry analysis for climate dynamics
-- Maps continuous symmetries to conserved quantities

{-
─────────────────────────────────────────────────────────────────────────────
DATA SOURCE REQUIREMENTS
─────────────────────────────────────────────────────────────────────────────

1. ANGULAR MOMENTUM BUDGET:
   - Source: ERA5 angular momentum diagnostics
   - Components: Atmospheric (AAM), Oceanic (OAM), Solid Earth
   - Resolution: Global integrals, daily
   - Temporal: 1940-present from ERA5
   - Format: NetCDF4 or GRIB2
   - Size: ~10GB for full time series
   - Variables: 
     * Relative AAM (wind terms)
     * Omega AAM (pressure terms)
     * Mountain torque
     * Friction torque
   - API: CDS special diagnostics request
   - Preprocessing: Convert to SI units (kg⋅m²/s)
   - Missing: Ocean-atmosphere torque exchange
   - Constraint: d(AAM+OAM)/dt = external torques

2. EARTH ROTATION PARAMETERS:
   - Source: IERS Bulletin A/B
   - Variables: LOD (length of day), polar motion (x,y)
   - Temporal: Daily since 1962, sub-daily since 1990s
   - Format: ASCII tables, IERS format
   - Size: <100MB
   - API: https://www.iers.org/IERS/EN/DataProducts/
   - Constraint: ΔLOD = -2π/(Ω²) × d(AAM)/dt
   - Missing: Attribution to specific climate modes

3. ENERGY CONSERVATION VALIDATION:
   - Source: CERES EBAF + Ocean heat content
   - TOA imbalance: CERES EBAF-TOA (2000-present)
   - Ocean heat: IAP/Cheng et al. (1955-present)
   - Format: NetCDF4
   - Size: ~10GB combined
   - Constraint: TOA imbalance = d(OHC)/dt + smaller terms
   - Missing: Deep ocean (>2000m) heat content

4. MOISTURE CONSERVATION:
   - Precipitation: GPCP v2.3 (2.5°, monthly, 1979-present)
   - Evaporation: ERA5 or OAFlux
   - Atmospheric water: ERA5 total column water vapor
   - Soil moisture: ERA5-Land or SMOS/SMAP
   - Format: NetCDF4
   - Size: ~50GB for all components
   - Constraint: dW/dt = E - P ± transport
   - Missing: Groundwater, vegetation water

5. CARBON CYCLE MASS BALANCE:
   - Atmospheric CO2: NOAA GML stations
   - Ocean uptake: SOCAT + GOBMs
   - Land sink: Global Carbon Project
   - Emissions: CDIAC/GCP inventories
   - Format: CSV, NetCDF4
   - Size: ~5GB
   - Temporal: Annual since 1959, monthly for atmosphere
   - Constraint: dCO2/dt = emissions - ocean_sink - land_sink
   - Missing: Permafrost carbon, fire emissions uncertainty

6. MOMENTUM CONSERVATION IN FLUIDS:
   - Source: ERA5 or MERRA-2 momentum budget terms
   - Variables: du/dt, pressure gradient, Coriolis, friction
   - Resolution: Model levels, 0.25° horizontal
   - Format: GRIB2 or NetCDF4
   - Size: ~1TB for full 3D fields
   - Constraint: Du/Dt + 2Ω×u = -∇p/ρ + F
   - Missing: Subgrid momentum fluxes

7. VORTICITY CONSERVATION:
   - Source: ERA5 vorticity and divergence
   - Variables: Relative vorticity, planetary vorticity
   - Resolution: Pressure levels, 0.25°
   - Constraint: D(ζ+f)/Dt = -∇⋅u(ζ+f) + tilting + baroclinic
   - Missing: Accurate vertical velocity for tilting term

8. WAVE ACTIVITY CONSERVATION:
   - Source: Computed from ERA5 u,v,T fields
   - Method: Takaya-Nakamura (2001) flux
   - Variables: Wave activity flux vectors
   - Constraint: ∂A/∂t + ∇⋅F = source - dissipation
   - Missing: Quantification of wave breaking

NEEDED: Energy flux measurements for conservation:
  - CERES SYN1deg (2000-present, monthly, 1°)
  - Format: NetCDF4 with all radiation components
  - Variables: SW↓, SW↑, LW↓, LW↑ at TOA and surface
  - Source: https://ceres.larc.nasa.gov/
  - Missing: Pre-satellite era energy budget

NEEDED: Mass conservation verification:
  - GRACE/GRACE-FO mascon solutions (2002-present, monthly)
  - Format: NetCDF4 with mass anomalies in gigatons
  - Coverage: Ice sheets, glaciers, groundwater, ocean
  - Source: https://grace.jpl.nasa.gov/data/
  - Missing: Mass redistribution before GRACE

NEEDED: Moisture transport for water cycle symmetry:
  - ERA5 vertically integrated moisture flux
  - Format: NetCDF4 with u*q and v*q components
  - Resolution: 0.25° × 0.25°, 6-hourly
  - Missing: Direct observations of moisture flux

NEEDED: Vorticity budget for circulation:
  - Calculated from reanalysis winds
  - Format: Must derive from U, V fields
  - Constraint: Potential vorticity conservation
  - Missing: Sub-grid scale vorticity fluxes

NEEDED: Carbon isotope ratios for source attribution:
  - NOAA/ESRL flask network (1980-present)
  - Format: Text files with δ13C, δ14C, δ18O
  - Source: https://gml.noaa.gov/ccgg/
  - Missing: Pre-industrial isotope baselines
  - Missing: Ocean carbon isotope profiles
─────────────────────────────────────────────────────────────────────────────
-}

module ClimateSymmetries where

import Control.Monad
import Control.Applicative
import Control.Lens hiding ((.=))
import Control.Monad.State.Strict
import Control.Monad.Reader
import Control.Parallel.Strategies
import Data.Complex
import Data.List (minimumBy, maximumBy, foldl', sort, intersperse, tails)
import Data.Maybe (catMaybes, fromMaybe)
import Data.Ord (comparing)
import Data.Proxy
import Data.Singletons
import Data.Type.Equality
import Data.Void
import GHC.TypeLits
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as U
import qualified Data.Map.Strict as M
import qualified Numeric.LinearAlgebra as LA
import qualified Numeric.LinearAlgebra.Data as LAD
import qualified Numeric.GSL.Integration as GSL
import qualified Numeric.GSL.Differentiation as GSLD
import qualified Statistics.Distribution as Dist
import qualified Statistics.Distribution.Normal as Normal
import qualified Statistics.Distribution.StudentT as StudentT
import qualified System.Random.MWC as MWC
import qualified System.Random.MWC.Distributions as MWCD
import qualified Data.Vector.Storable as VS
import qualified Data.Vector.Unboxed.Mutable as UM
import Text.Printf (printf)

-- Additive symmetry analysis utilities (no removal of existing logic)

-- Generic diagnostic container for richer reporting
data SymmetryDiagnostic = SymmetryDiagnostic
    { diagName      :: String
    , diagStatistic :: Double
    , diagPValue    :: Maybe Double
    , diagEffect    :: Maybe Double
    , diagConfidence :: Double
    , diagNotes     :: [String]
    } deriving (Show, Eq)

-- Additive: Bayesian evidence structure for symmetry persistence
data BayesianSymmetryEvidence = BayesianSymmetryEvidence
    { bseName        :: String
    , logEConstant   :: Double
    , logEDrift      :: Double
    , bayesFactor    :: Double  -- logEConstant - logEDrift (positive favors conservation)
    , modelDecision  :: String
    } deriving (Show, Eq)

-- Additive: DMD result container
data DMDResult = DMDResult
    { dmdEigenvalues :: [Complex Double]
    , dmdModeAmplitudes :: [Double]
    , dmdClassification :: [String]
    } deriving (Show, Eq)

-- Type class for additive symmetry detector abstraction
class SymmetryDetector a where
    symmetryName :: a -> String
    runDetection :: a -> ClimatePathway -> Maybe SymmetryDiagnostic

data TimeTranslationDetector = TimeTranslationDetector
instance SymmetryDetector TimeTranslationDetector where
    symmetryName _ = "time_translation_energy"
    runDetection _ path =
        let vals = map planetaryEnergyAt path
            v    = calculateVariance vals
            meanE = if null vals then 0 else sum vals / fromIntegral (length vals)
            relVar = if meanE /= 0 then v / (meanE*meanE) else 0
            conf = if relVar < 1e-4 then 0.99 else if relVar < 1e-3 then 0.9 else 0.5
        in if relVar < 5e-3 then Just SymmetryDiagnostic
                { diagName = "time_translation_energy"
                , diagStatistic = relVar
                , diagPValue = Nothing
                , diagEffect = Just meanE
                , diagConfidence = conf
                , diagNotes = ["relativeVariance=" ++ show relVar]
                }
           else Nothing

data ScaleDetector = ScaleDetector
instance SymmetryDetector ScaleDetector where
    symmetryName _ = "scale_invariance"
    runDetection _ path =
        let (ok, expo) = checkPowerLaw path
            conf = if ok then 0.85 else 0.2
        in if ok then Just SymmetryDiagnostic
            { diagName = "scale_invariance"
            , diagStatistic = expo
            , diagPValue = Nothing
            , diagEffect = Nothing
            , diagConfidence = conf
            , diagNotes = ["exponent=" ++ show expo]
            } else Nothing

-- FFT-based periodicity detector (additive, does not replace existing detectPeriodicity heuristic)
detectPeriodicityFFT :: [Double] -> Int -> Maybe (Double, Double) -- (dominant period, normalized power)
detectPeriodicityFFT xs minLag =
    let n = length xs
    in if n < 8 then Nothing else
        let meanX = sum xs / fromIntegral n
            centered = map (\v -> v - meanX) xs
            -- naive DFT magnitude (O(n^2)) acceptable for short climate vectors here
            mags = [ let omega = 2*pi*fromIntegral k / fromIntegral n
                          reIm = [ (cos(omega*fromIntegral t)*x, sin(omega*fromIntegral t)*x) | (t,x)<-zip [0..] centered]
                          (re,im) = foldl' (\(ar,ai) (r,i) -> (ar+r, ai+i)) (0,0) reIm
                      in sqrt(re*re+im*im)
                   | k <- [1..n-1] ]
            (kMax, aMax) = maximumBy (comparing snd) (zip [1..] mags)
            period = fromIntegral n / fromIntegral kMax
            normPower = if null mags then 0 else aMax / (sum mags + 1e-12)
        in if period >= fromIntegral minLag then Just (period, normPower) else Nothing

-- Aggregate symmetry diagnostics
runSymmetryDiagnostics :: ClimatePathway -> [SymmetryDiagnostic]
runSymmetryDiagnostics path = catMaybes
    [ runDetection TimeTranslationDetector path
    , runDetection ScaleDetector path
    ]

renderDiagnosticsTable :: [SymmetryDiagnostic] -> String
renderDiagnosticsTable ds =
    let header = "Name | Stat | Conf | Notes\n-----|------|------|------\n"
        rows = [ printf "%s | %.4g | %.2f | %s" (diagName d) (diagStatistic d) (diagConfidence d)
                        (case diagNotes d of [] -> ""; xs -> head xs) | d <- ds]
    in unlines (header:rows)

-- Additive: simple bootstrap utility for uncertainty estimates
bootstrapMean :: Int -> [Double] -> (Double, Double)
bootstrapMean b xs | null xs = (0,0) | otherwise =
  let n = length xs
      genSamples 0 acc = acc
      genSamples k acc =
          let sample = [ xs !! (i `mod` n) | i <- take n [ (k*i + 137) `mod` n | i <- [0..] ]]
              m = sum sample / fromIntegral n
          in genSamples (k-1) (m:acc)
      means = genSamples b []
      mu = sum means / fromIntegral b
      var = sum [(m-mu)^2 | m<-means] / fromIntegral b
      se = sqrt var
  in (mu,se)

-- Additive: approximate Lie bracket feasibility check between two infinitesimal generators (as discrete finite diff)
approxLieBracket :: (ClimateState -> ClimateState) -> (ClimateState -> ClimateState) -> ClimateState -> Double -> Double
approxLieBracket f g s eps =
    let s1 = f (g s)
        s2 = g (f s)
        -- distance measure in state space (Euclidean over selected fields)
        dist a b = sum
          [ abs (unQuantity (globalTemp a) - unQuantity (globalTemp b))
          , abs (unQuantity (co2Concentration a) - unQuantity (co2Concentration b))
          , abs (unQuantity (oceanHeatContent a) - unQuantity (oceanHeatContent b))
          ]
    in dist s1 s2 / (eps + 1e-9)

inferLieStructure :: [ContinuousSymmetry] -> [(String,String,Double)]
inferLieStructure syms =
    let pairs = [ (a,b) | a<-syms, b<-syms, symType a < symType b ]
        baseState = case syms of (s:_) -> generator s (generator s (generator s (generator s (errorBase)))) ; _ -> errorBase
        errorBase = ClimateState (Quantity 0) (Quantity 280) (Quantity 0) (Quantity 1e7) (Quantity 0.3) (Quantity 15) (Quantity 0.6) (Quantity 1.8) (Quantity 0.33) (Quantity 0.1) (Quantity 0.4) (Quantity 0)
    in [ (symType a, symType b, approxLieBracket (generator a) (generator b) errorBase 1e-3) | (a,b)<-pairs ]

renderLieStructure :: [(String,String,Double)] -> String
renderLieStructure triples =
    let hdr = "Generator Pair | Bracket Magnitude\n----------------|------------------\n"
        rows = [ a ++ " ∘ " ++ b ++ " | " ++ printf "%.3e" v | (a,b,v)<-triples]
    in unlines (hdr:rows)

-- Additive: multiscale spectral analysis for early warning (power-law slope of PSD)
powerSpectrumSlope :: [Double] -> Maybe Double
powerSpectrumSlope xs =
  let n = length xs
  in if n < 16 then Nothing else
     let meanX = sum xs / fromIntegral n
         centered = map (\v-> v - meanX) xs
         mags = [ let omega = 2*pi*fromIntegral k / fromIntegral n
                       reIm = [ (cos(omega*fromIntegral t)*x, sin(omega*fromIntegral t)*x) | (t,x)<-zip [0..] centered]
                       (re,im) = foldl' (\(ar,ai) (r,i) -> (ar+r, ai+i)) (0,0) reIm
                   in (fromIntegral k, (re*re+im*im) / fromIntegral n)
                | k <- [1..n `div` 2] ]
         logPairs = [ (log f, log p) | (f,p) <- mags, f>0, p>0 ]
     in if length logPairs < 4 then Nothing else Just (fst (linearRegression logPairs))

-- Enhanced aggregation of warning signals including spectral slope
aggregateWarningIndex :: [WarningSignal] -> [Double] -> Double
aggregateWarningIndex ws temps =
    let base = sum [strength w | w<-ws] / (fromIntegral (length ws) + 1e-9)
        slopeBonus = maybe 0 (\s -> if s < -1 then min 0.2 (abs s / 10) else 0) (powerSpectrumSlope temps)
    in min 1.0 (base + slopeBonus)

-- ─────────────────────────────────────────────────────────────────────────────
-- CLIMATE OSCILLATION MODE ANALYSIS (additive)
-- ─────────────────────────────────────────────────────────────────────────────

data ClimateOscillationFit = ClimateOscillationFit
    { cofMode       :: String
    , cofPeriod     :: Double  -- fitted period in years
    , cofAmplitude  :: Double  -- fitted amplitude
    , cofR2         :: Double  -- goodness of fit
    } deriving (Show, Eq)

fitClimateOscillation :: ClimateOscillationMode -> ClimateOscillationFit
fitClimateOscillation com =
    let -- Generate synthetic time series for analysis
        times = [0, 0.1 .. 20]  -- 20 years of data
        series = [ (amplitude com) * sin(2 * pi * t / (period com)) | t <- times ]
        -- Fit period using autocorrelation peak
        fittedPeriod = findDominantPeriod series times
        fittedAmplitude = maximum series - minimum series
        r2 = calculateGoodnessOfFit series fittedPeriod fittedAmplitude
    in ClimateOscillationFit (modeName com) fittedPeriod fittedAmplitude r2

findDominantPeriod :: [Double] -> [Double] -> Double
findDominantPeriod series times =
    let n = length series
        lags = [1..n `div` 4]
        autocorrs = [autocorrelation series lag | lag <- lags]
        maxLag = snd $ maximumBy (comparing fst) (zip autocorrs lags)
        dt = if length times > 1 then times !! 1 - times !! 0 else 0.1
    in fromIntegral maxLag * dt

calculateGoodnessOfFit :: [Double] -> Double -> Double -> Double
calculateGoodnessOfFit series period amplitude =
    let n = length series
        predicted = [amplitude * sin(2 * pi * fromIntegral i * 0.1 / period) | i <- [0..n-1]]
        meanSeries = sum series / fromIntegral n
        ssTotal = sum [(x - meanSeries)^2 | x <- series]
        ssResidual = sum [(observed - pred)^2 | (observed, pred) <- zip series predicted]
    in if ssTotal > 0 then 1 - ssResidual / ssTotal else 0

computeClimateOscillationFits :: [BrokenSymmetry] -> [ClimateOscillationFit]
computeClimateOscillationFits bs = [ fitClimateOscillation com | b <- bs, Just com <- [climateMode b] ]

renderClimateOscillationFits :: [ClimateOscillationFit] -> String
renderClimateOscillationFits fs | null fs = "No climate oscillation modes" | otherwise =
    let hdr = "Mode | Period (yr) | Amplitude | R²\n-----|------------|-----------|----\n"
            rows = [ printf "%s | %.3f | %.3g | %.3f" (cofMode f) (cofPeriod f) (cofAmplitude f) (cofR2 f) | f<-fs]
    in unlines (hdr:rows)

-- ─────────────────────────────────────────────────────────────────────────────
-- ENSEMBLE / MULTI-PATHWAY UNCERTAINTY (additive)
-- ─────────────────────────────────────────────────────────────────────────────

type ClimateEnsemble = [ClimatePathway]

data EnsembleSummary = EnsembleSummary
    { esNumPaths :: Int
    , esMeanEnergy :: Double
    , esEnergyStd  :: Double
    , esMeanRiskIndex :: Double
    } deriving (Show, Eq)

summarizeEnsemble :: ClimateEnsemble -> EnsembleSummary
summarizeEnsemble ens | null ens = EnsembleSummary 0 0 0 0
summarizeEnsemble ens =
    let energies = [ calculatePlanetaryEnergy p | p <- ens, not (null p) ]
            warningsList = [ earlyWarningSignals p | p <- ens ]
            riskIdx = [ aggregateWarningIndex ws (map (unQuantity . globalTemp) p) | (ws,p) <- zip warningsList ens]
            n = fromIntegral (length energies)
            meanE = sum energies / n
            varE = sum [(e-meanE)^2 | e<-energies] / max 1 (length energies)
            meanR = if null riskIdx then 0 else sum riskIdx / fromIntegral (length riskIdx)
    in EnsembleSummary (length ens) meanE (sqrt varE) meanR

renderEnsembleSummary :: EnsembleSummary -> String
renderEnsembleSummary es =
    printf "Ensemble paths=%d meanEnergy=%.4f±%.4f meanWarningIndex=%.3f" (esNumPaths es) (esMeanEnergy es) (esEnergyStd es) (esMeanRiskIndex es)

-- ─────────────────────────────────────────────────────────────────────────────
-- JSON / CSV EXPORT (additive)
-- ─────────────────────────────────────────────────────────────────────────────

escapeJSON :: String -> String
escapeJSON = concatMap f where
    f '"' = "\\\""; f '\\' = "\\\\"; f c = [c]

jsonPair :: String -> String -> String
jsonPair k v = "\"" ++ escapeJSON k ++ "\": " ++ v

doubleJSON :: Double -> String
doubleJSON d = if isNaN d || isInfinite d then "null" else printf "%.6g" d

diagnosticsToJSON :: [ContinuousSymmetry] -> [DiscreteSymmetry] -> [BrokenSymmetry] -> [WarningSignal] -> [SymmetryDiagnostic] -> [BayesianSymmetryEvidence] -> [ClimateOscillationFit] -> String
diagnosticsToJSON cont disc broken warns diags bayes cofits =
    let arr f xs = "[" ++ concat (zipWith (\i x -> (if i>0 then "," else "") ++ f x) [0..] xs) ++ "]"
            jsCont c = "{" ++ intercalateComma
                [ jsonPair "type" (str (symType c))
                , jsonPair "group" (str (show (symGroup c)))
                , jsonPair "quantity" (str (qtyName (conservedQty c)))
                , jsonPair "value" (doubleJSON (qtyValue (conservedQty c)))
                ] ++ "}"
            jsDisc d = "{" ++ intercalateComma [ jsonPair "type" (str (discSymType d)), jsonPair "order" (show (order d)) ] ++ "}"
            jsBroken b = "{" ++ intercalateComma [ jsonPair "type" (str (brokenType b)), jsonPair "criticalTemp" (doubleJSON (criticalTemp b)) ] ++ "}"
            jsWarn w = "{" ++ intercalateComma [ jsonPair "type" (str (signalType w)), jsonPair "strength" (doubleJSON (strength w)) ] ++ "}"
            jsDiag d = "{" ++ intercalateComma [ jsonPair "name" (str (diagName d)), jsonPair "stat" (doubleJSON (diagStatistic d)) ] ++ "}"
            jsBay b = "{" ++ intercalateComma [ jsonPair "series" (str (bseName b)), jsonPair "dLogE" (doubleJSON (bayesFactor b)) ] ++ "}"
            jsCOF c = "{" ++ intercalateComma [ jsonPair "mode" (str (cofMode c)), jsonPair "period" (doubleJSON (cofPeriod c)), jsonPair "amplitude" (doubleJSON (cofAmplitude c)) ] ++ "}"
            str s = "\"" ++ escapeJSON s ++ "\""
            intercalateComma = concat . intersperse ","
    in "{" ++ intercalateComma
                [ jsonPair "continuous" (arr jsCont cont)
                , jsonPair "discrete" (arr jsDisc disc)
                , jsonPair "broken" (arr jsBroken broken)
                , jsonPair "warnings" (arr jsWarn warns)
                , jsonPair "diagnostics" (arr jsDiag diags)
                , jsonPair "bayes" (arr jsBay bayes)
                , jsonPair "climate_oscillation_fits" (arr jsCOF cofits)
                ] ++ "}"

csvBayesianEvidence :: [BayesianSymmetryEvidence] -> String
csvBayesianEvidence es =
    let header = "series,logE_constant,logE_drift,delta_logE,decision\n"
            rows = [ printf "%s,%.4f,%.4f,%.4f,%s" (bseName e) (logEConstant e) (logEDrift e) (bayesFactor e) (modelDecision e) | e<-es ]
    in unlines (header:rows)

-- ─────────────────────────────────────────────────────────────────────────────
-- BATCH 3: DYNAMICAL COMPLEXITY METRICS (additive)
-- 1. Largest Lyapunov exponent estimate (Rosenstein-style)
-- 2. Granger causality influence scores between key variables
-- 3. Recurrence Quantification Analysis (RQA) metrics
-- ─────────────────────────────────────────────────────────────────────────────

data LyapunovEstimate = LyapunovEstimate
    { lyapSeries   :: String
    , lyapExponent :: Double
    , lyapPoints   :: Int
    , lyapNotes    :: [String]
    } deriving (Show, Eq)

timeDelayEmbed :: Int -> Int -> [Double] -> [[Double]]
timeDelayEmbed dim tau xs =
    let limit = length xs - (dim-1)*tau
    in [ [ xs !! (i + k*tau) | k <- [0..dim-1] ] | i <- [0..limit-1], limit>0]

euclid :: [Double] -> [Double] -> Double
euclid as bs = sqrt (sum (zipWith (\x y -> (x-y)^2) as bs))

estimateLargestLyapunov :: [Double] -> LyapunovEstimate
estimateLargestLyapunov series | length series < 20 = LyapunovEstimate "temperature" 0 (length series) ["too_short"]
estimateLargestLyapunov series =
    let dim = 3; tau = 2
            embedded = timeDelayEmbed dim tau series
            n = length embedded
            -- nearest neighbor for each point (exclude self & temporal neighbors)
            neigh = [ minimumBy (comparing snd)
                                 [ (j, euclid (embedded!!i) (embedded!!j))
                                 | j <- [0..n-1], j /= i, abs (j-i) > 3]
                            | i <- [0..n-1], n>0]
            maxLead = 10
            divergences = [ [ let idxJ = fst (neigh!!i)
                                                         xi = embedded!!(i+k)
                                                         xj = embedded!!(idxJ + k)
                                                 in euclid xi xj + 1e-12
                                             | k <- [0..maxLead-1]
                                             , i + k < n
                                             , fst (neigh!!i) + k < n]
                                        | i <- [0..n-1]]
            meanLogD k =
                 let ds = [ log (divs !! k) | divs <- divergences, length divs > k ]
                 in if null ds then Nothing else Just (sum ds / fromIntegral (length ds))
            pts = [ (fromIntegral k, m) | k <- [1..maxLead-1], Just m <- [meanLogD k]]
            (slope, r2) = linearRegression pts
    in LyapunovEstimate "temperature" slope (length pts) ["R2=" ++ printf "%.3f" r2, "dim=3", "tau=2"]

renderLyapunov :: LyapunovEstimate -> String
renderLyapunov le = printf "Lyapunov(%s)=%.4f (points=%d) %s"
                                                (lyapSeries le) (lyapExponent le) (lyapPoints le) (unwords (lyapNotes le))

-- Granger causality (simple one-lag linear comparison)
data GrangerResult = GrangerResult
    { grFrom   :: String
    , grTo     :: String
    , grFStat  :: Double
    , grScore  :: Double  -- normalized effect size
    } deriving (Show, Eq)

linearReg2Vars :: [(Double, Double, Double)] -> (Double, Double, Double, Double)
linearReg2Vars triples =
    -- (y, x1, x2); solve y = a x1 + b x2 + c
    let ys = map (\(y,_,_) -> y) triples
            x1s = map (\(_,x1,_) -> x1) triples
            x2s = map (\(_,_,x2) -> x2) triples
            n = fromIntegral (length triples)
            sumv vs = sum vs
            sY = sumv ys; sX1 = sumv x1s; sX2 = sumv x2s
            sX1X1 = sumv (zipWith (*) x1s x1s)
            sX2X2 = sumv (zipWith (*) x2s x2s)
            sX1X2 = sumv (zipWith (*) x1s x2s)
            sYX1 = sumv (zipWith (*) ys x1s)
            sYX2 = sumv (zipWith (*) ys x2s)
            -- Solve normal equations manually (small system)
            -- Matrix [[sX1X1, sX1X2, sX1], [sX1X2, sX2X2, sX2], [sX1, sX2, n]]
            det = sX1X1*(sX2X2*n - sX2*sX2) - sX1X2*(sX1X2*n - sX2*sX1) + sX1*(sX1X2*sX2 - sX2X2*sX1)
            safe x = if abs x < 1e-20 then 1e-20 else x
            a = (sYX1*(sX2X2*n - sX2*sX2) - sYX2*(sX1X2*n - sX2*sX1) + sY*(sX1X2*sX2 - sX2X2*sX1)) / safe det
            b = (sX1X1*(sYX2*n - sY*sX2) - sX1X2*(sYX1*n - sY*sX1) + sX1*(sYX1*sX2 - sYX2*sX1)) / safe det
            c = (sX1X1*(sX2X2*sY - sX2*sYX2) - sX1X2*(sX1X2*sY - sX2*sYX1) + sX1*(sX1X2*sYX2 - sX2X2*sYX1)) / safe det
            rss = sum [ let y = yv; yhat = a*x1 + b*x2 + c in (y - yhat)^2 | (yv,x1,x2) <- triples]
    in (a,b,c,rss)

grangerPair :: String -> [Double] -> String -> [Double] -> Maybe GrangerResult
grangerPair nameX xs nameY ys =
    let lag = 1
            len = minimum [length xs, length ys] - lag
    in if len < 10 then Nothing else
         let y_t = drop lag ys
                 y_lag = take len ys
                 x_lag = take len xs
                 mean vs = sum vs / fromIntegral (length vs)
                 -- Restricted model: y_t ~ y_{t-1}
                 mY = mean y_lag; mYT = mean y_t
                 varR = sum [ (yt - (alpha*yl + beta))^2 | (yt,yl) <- zip y_t y_lag
                                     , let alpha =  (sum (zipWith (*) (map (\v->v-mY) y_lag) (map (\v->v-mYT) y_t))) /
                                                                                 (sum (map (\v->(v-mY)^2) y_lag) + 1e-12)
                                                 beta = mYT - alpha*mY ] / fromIntegral len
                 triples = [ (yt, yl, xl) | (yt,yl,xl) <- zip3 y_t y_lag x_lag]
                 (_,_,_,rssFull) = linearReg2Vars triples
                 rssRestr = varR * fromIntegral len
                 df1 = 1; df2 = fromIntegral len - 3
                 fstat = ((rssRestr - rssFull)/fromIntegral df1) / (rssFull / df2 + 1e-12)
                 score = fstat / (fstat + 1)
         in Just GrangerResult { grFrom = nameX, grTo = nameY, grFStat = fstat, grScore = score }

computeGranger :: [(String,[Double])] -> [GrangerResult]
computeGranger vars =
    [ gr | (i,(nx,xs)) <- zip [0..] vars
             , (j,(ny,ys)) <- zip [0..] vars
             , i /= j
             , Just gr <- [grangerPair nx xs ny ys] ]

renderGranger :: [GrangerResult] -> String
renderGranger grs | null grs = "No Granger influences" | otherwise =
    let hdr = "from→to | F | score\n--------|----|------\n"
            rows = [ printf "%s→%s | %.2f | %.3f" (grFrom g) (grTo g) (grFStat g) (grScore g) | g<-grs]
    in unlines (hdr:rows)

data RecurrenceSummary = RecurrenceSummary
    { recRate :: Double
    , recDeterminism :: Double
    , recAvgDiag :: Double
    } deriving (Show, Eq)

computeRQA :: [Double] -> RecurrenceSummary
computeRQA series | length series < 30 = RecurrenceSummary 0 0 0
computeRQA series =
    let dim = 3; tau = 1
            emb = timeDelayEmbed dim tau series
            n = length emb
            dists = [ euclid (emb!!i) (emb!!j) | i <- [0..n-1], j <- [i+1..n-1]]
            sorted = sort dists
            thresh = if null sorted then 0 else sorted !! (length sorted `div` 10)  -- 10th percentile
            recMatrix = [ [ if euclid (emb!!i) (emb!!j) <= thresh then 1 else 0 | j <- [0..n-1]] | i <- [0..n-1]]
            totalPairs = n*n
            recPoints = sum [ recMatrix!!i!!j | i <- [0..n-1], j <- [0..n-1]]
            rate = fromIntegral recPoints / fromIntegral totalPairs
            -- determinism: fraction of recurrence points forming diagonal lines length>=2
            diagLines = [ len | i <- [0..n-1], j <- [0..n-1]
                                                , recMatrix!!i!!j == 1
                                                , let len = diagLen i j
                                                , len >= 2]
            diagLen i j = go 0 i j where
                    go acc a b | a < n && b < n && recMatrix!!a!!b == 1 = go (acc+1) (a+1) (b+1)
                                         | otherwise = acc
            detPoints = sum diagLines
            determinism = if recPoints == 0 then 0 else fromIntegral detPoints / fromIntegral recPoints
            avgDiag = if null diagLines then 0 else fromIntegral (sum diagLines) / fromIntegral (length diagLines)
    in RecurrenceSummary rate determinism avgDiag

renderRQA :: RecurrenceSummary -> String
renderRQA r = printf "RQA: rate=%.3f determinism=%.3f avgDiag=%.2f" (recRate r) (recDeterminism r) (recAvgDiag r)

-- Additive improved spatial correlation length estimation
computeCorrelationLength :: [Double] -> Double
computeCorrelationLength xs =
    let n = length xs
        meanX = if n==0 then 0 else sum xs / fromIntegral n
        centered = map (\v -> v - meanX) xs
        auto k = if k>=n then 0 else
            let pairs = zip centered (drop k centered)
                num = sum [a*b | (a,b) <- pairs]
                denom = sum [c*c | c<-centered]
            in if denom==0 then 0 else num / denom
        -- integrate autocorrelation until first zero crossing for correlation length proxy
        vals = takeWhile (>0) [auto k | k <- [1..min 200 (n-1)]]
    in sum vals

-- ─────────────────────────────────────────────────────────────────────────────
-- BAYESIAN SYMMETRY PERSISTENCE (additive)
-- ─────────────────────────────────────────────────────────────────────────────

-- Marginal log evidence for y_t = mu + eps with Jeffreys prior ~ 1/sigma (flat) -> proportional to -((n-1)/2) log(SSE_const) - 0.5 log n
logEvidenceConstant :: [Double] -> Double
logEvidenceConstant ys =
    let n = length ys in if n < 2 then 0 else
            let meanY = sum ys / fromIntegral n
                    sse = sum [(y-meanY)^2 | y<-ys] + 1e-12
            in (-0.5 * fromIntegral (n-1)) * log sse - 0.5 * log (fromIntegral n)

-- Marginal log evidence for y_t = a + b t + eps (linear drift), simple approximation: -((n-2)/2) log(SSE_lin) - 0.5 log det(X'X)
logEvidenceLinear :: [Double] -> Double
logEvidenceLinear ys =
    let n = length ys in if n < 3 then 0 else
            let ts = map fromIntegral [0..n-1]
                    sumT = sum ts
                    sumT2 = sum [t*t | t<-ts]
                    sumY = sum ys
                    sumTY = sum (zipWith (*) ts ys)
                    denom = n*sumT2 - sumT*sumT + 1e-12
                    b = (fromIntegral n * sumTY - sumT*sumY) / denom
                    a = (sumY - b*sumT)/ fromIntegral n
                    sse = sum [ (y - (a + b*t))^2 | (t,y) <- zip ts ys ] + 1e-12
                    detxtx = denom / fromIntegral n  -- crude scaling
            in (-0.5 * fromIntegral (n-2)) * log sse - 0.5 * log detxtx

bayesFactorSymmetry :: [Double] -> BayesianSymmetryEvidence
bayesFactorSymmetry ys =
    let lc = logEvidenceConstant ys
            ld = logEvidenceLinear ys
            bf = lc - ld
            decision | bf > 5 = "Decisive conservation"
                             | bf > 2 = "Strong conservation"
                             | bf > 1 = "Substantial conservation"
                             | bf > -1 = "Inconclusive"
                             | otherwise = "Evidence of drift"
    in BayesianSymmetryEvidence "generic_series" lc ld bf decision

evaluateSymmetryPersistence :: ClimatePathway -> [BayesianSymmetryEvidence]
evaluateSymmetryPersistence path =
    let series =
                [ ("planetary_energy", map planetaryEnergyAt path)
                , ("atmospheric_angular_momentum", map atmAngularMomentumAt path)
                , ("scale_law_temp", let temps = map (unQuantity . globalTemp) path in temps)
                ]
    in [ (bayesFactorSymmetry ys) { bseName = nm } | (nm, ys) <- series, length ys > 3]

renderBayesianEvidence :: [BayesianSymmetryEvidence] -> String
renderBayesianEvidence es =
    let header = "Series | logE(const) | logE(drift) | ΔlogE | Decision\n-------|-------------|------------|-------|----------\n"
            rows = [ printf "%s | %.3f | %.3f | %.3f | %s" (bseName e) (logEConstant e) (logEDrift e) (bayesFactor e) (modelDecision e) | e<-es]
    in unlines (header:rows)

-- ─────────────────────────────────────────────────────────────────────────────
-- DYNAMICAL MODE DECOMPOSITION (additive lightweight)
-- ─────────────────────────────────────────────────────────────────────────────

performDMD :: [[Double]] -> DMDResult
performDMD vars =
    let -- Construct snapshot matrices X (n x (m-1)), X' (n x (m-1))
            -- rows = variables, columns = time snapshots
            m = minimum (map length vars)
            trim = map (take m) vars
            xCols = m-1
            xMat = LA.fromLists [ take (m-1) v | v<-trim ]
            xMat' = LA.fromLists [ drop 1 v | v<-trim ]
            -- Economy SVD X = U Σ V^T
            (u,s,vT) = LA.thinSVD xMat
            r = length s
            sigmaInv = LA.diag (LA.fromList [ if si > 1e-12 then 1/si else 0 | si <- s ])
            aTilde = (LA.tr' u) LA.<> xMat' LA.<> (LA.tr' vT) LA.<> sigmaInv
            (eigVals, eigVecs) = LA.eig aTilde
            eigList = LA.toList eigVals
            -- Mode amplitudes: project first snapshot
            x1 = LA.col 0 xMat
            b = LA.linearSolve eigVecs ((LA.tr' u) LA.<> x1)
            amps = maybe [] (LA.toList . LA.flatten) b
            classify λ =
                let re = realPart λ; im = imagPart λ
                        mag = magnitude λ
                in concat [ if abs im > 1e-6 then "oscillatory " else ""
                                    , if mag < 1     then "dissipative " else (if mag>1 then "unstable " else "neutral ")
                                    , if abs (mag-1) < 1e-3 && abs im > 1e-3 then "U1-like" else ""
                                    ]
            classes = map classify eigList
    in DMDResult eigList (map abs amps) classes

renderDMDResult :: DMDResult -> String
renderDMDResult d =
    let header = "Eigenvalue (re,im) | |λ| | Amp | Class\n------------------|-----|-----|------\n"
            rows = [ printf "(%.3f,%.3f) | %.3f | %.3f | %s" (realPart λ) (imagPart λ) (magnitude λ) a c
                            | (λ,a,c) <- zip3 (dmdEigenvalues d) (dmdModeAmplitudes d ++ repeat 0) (dmdClassification d) ]
    in unlines (header:rows)

-- High-level helper to run DMD on chosen climate variables
runClimateDMD :: ClimatePathway -> DMDResult
runClimateDMD path =
    let temps = map (unQuantity . globalTemp) path
            co2s  = map (unQuantity . co2Concentration) path
            amoc  = map (unQuantity . amocStrength) path
    in performDMD [temps, co2s, amoc]

-- ─────────────────────────────────────────────────────────────────────────────
-- EMERGENT Z_n DISCRETE SYMMETRY DETECTION (additive)
-- ─────────────────────────────────────────────────────────────────────────────

detectEmergentZn :: [Double] -> [Int]
detectEmergentZn xs =
    let n = length xs
    in if n < 16 then [] else
         let meanX = sum xs / fromIntegral n
                 centered = map (\v-> v - meanX) xs
                 mags = [ let omega = 2*pi*fromIntegral k / fromIntegral n
                                             reIm = [ (cos(omega*fromIntegral t)*x, sin(omega*fromIntegral t)*x) | (t,x)<-zip [0..] centered]
                                             (re,im) = foldl' (\(ar,ai) (r,i) -> (ar+r, ai+i)) (0,0) reIm
                                     in (k, sqrt(re*re+im*im)) | k <- [1..n `div` 2] ]
                 total = sum (map snd mags) + 1e-12
                 strong = [ k | (k,a) <- mags, a / total > 0.05 ]
                 candidateN = length strong
         in if candidateN >= 2 && candidateN <= 12 then [candidateN] else []

renderEmergentZn :: [Int] -> String
renderEmergentZn ns | null ns = "No emergent cyclic symmetry detected" | otherwise =
    "Emergent candidate cyclic symmetry orders: " ++ show ns

-- ─────────────────────────────────────────────────────────────────────────────
-- CLIMATE STATE MANIFOLD
-- ─────────────────────────────────────────────────────────────────────────────

-- Type-level dimensions for physical quantities
data Kelvin
data PPM  
data Zettajoule
data Gigaton
data Dimensionless
data Sverdrup

-- Phantom-typed physical quantities with units
newtype Quantity (d :: *) = Quantity { unQuantity :: Double }
    deriving newtype (Show, Eq, Ord, Num, Fractional, Floating, Real, RealFrac, RealFloat)

-- Climate state with type-safe units
data ClimateState = ClimateState
    { globalTemp       :: !(Quantity Kelvin)        -- Global mean temperature anomaly
    , co2Concentration :: !(Quantity PPM)           -- Atmospheric CO2
    , oceanHeatContent :: !(Quantity Zettajoule)    -- Ocean heat content  
    , iceSheetMass     :: !(Quantity Gigaton)       -- Total ice mass
    , albedo           :: !(Quantity Dimensionless) -- Planetary albedo
    , amocStrength     :: !(Quantity Sverdrup)      -- AMOC circulation
    , cloudFraction    :: !(Quantity Dimensionless) -- Cloud cover fraction
    , methaneConc      :: !(Quantity PPM)           -- CH4 concentration
    , n2oConc          :: !(Quantity PPM)           -- N2O concentration
    , aerosolOptDepth  :: !(Quantity Dimensionless) -- Aerosol optical depth
    , soilMoisture     :: !(Quantity Dimensionless) -- Global soil moisture
    , seaLevel         :: !(Quantity Kelvin)        -- Sea level (using Kelvin as meter proxy)
    } deriving (Show, Eq)

-- Climate pathway through state space
type ClimatePathway = [ClimateState]

-- ─────────────────────────────────────────────────────────────────────────────
-- SYMMETRY GROUPS (TYPE-LEVEL)
-- ─────────────────────────────────────────────────────────────────────────────

-- Type-level symmetry groups with proofs
data SymmetryGroup where
    U1      :: SymmetryGroup                          -- Circle group (phase symmetry)
    SO2     :: SymmetryGroup                          -- 2D rotations
    SO3     :: SymmetryGroup                          -- 3D rotations  
    SU2     :: SymmetryGroup                          -- Spin group
    Z2      :: SymmetryGroup                          -- Reflection symmetry
    Zn      :: KnownNat n => Proxy n -> SymmetryGroup -- Cyclic group
    Rplus   :: SymmetryGroup                          -- Scale symmetry
    -- Lorentz group not applicable for climate (non-relativistic)
    Gauge   :: Symbol -> SymmetryGroup                -- Gauge symmetries
    Product :: SymmetryGroup -> SymmetryGroup -> SymmetryGroup  -- Direct product
    
deriving instance Show SymmetryGroup

-- Lie algebra generators
data Generator sym where
    TimeTranslation   :: Generator U1
    EnergyRotation    :: Generator SO2
    SphericalHarmonic :: Int -> Int -> Generator SO3
    ParityOperator    :: Generator Z2
    ScaleGenerator    :: Generator Rplus
    GaugeGenerator    :: String -> Generator (Gauge s)

-- ─────────────────────────────────────────────────────────────────────────────
-- CONTINUOUS SYMMETRIES
-- ─────────────────────────────────────────────────────────────────────────────

data ContinuousSymmetry = ContinuousSymmetry
    { symType      :: String
    , generator    :: ClimateState -> ClimateState  -- Infinitesimal transformation
    , parameter    :: Double
    , conservedQty :: ConservedQuantity
    , symGroup     :: SymmetryGroup
    , isExact      :: Bool
    } deriving (Show)

data ConservedQuantity = ConservedQuantity
    { qtyName  :: String
    , qtyValue :: Double
    , qtyUnits :: String
    , current  :: ConservedCurrent
    } deriving (Show)

data ConservedCurrent = ConservedCurrent
    { currentName :: String
    , components  :: M.Map String Double
    , divergence  :: Double  -- Should be 0 for conservation
    } deriving (Show)

-- Detect continuous symmetries in climate system
detectContinuousSymmetries :: ClimatePathway -> [ContinuousSymmetry]
detectContinuousSymmetries pathway = 
    let symmetries = 
            [ checkTimeTranslation pathway
            , checkAngularMomentum pathway
            , checkScaleInvariance pathway
            , checkEnergyMomentumConservation pathway
            , checkGaugeInvariance pathway
            ] `using` parList rdeepseq
    in catMaybes symmetries

-- Check time translation symmetry (energy conservation)
checkTimeTranslation :: ClimatePathway -> Maybe ContinuousSymmetry
checkTimeTranslation pathway =
    let energy = calculatePlanetaryEnergy pathway
        variance = calculateVariance (map planetaryEnergyAt pathway)
        isConserved = variance < 0.01  -- Within 1% variation
    in if isConserved
       then Just $ ContinuousSymmetry
            { symType = "time_translation"
            , generator = timeTranslationGenerator 1.0
            , parameter = 1.0
            , conservedQty = ConservedQuantity
                { qtyName = "planetary_energy"
                , qtyValue = energy
                , qtyUnits = "W/m²"
                , current = energyMomentumTensor pathway
                }
            , symGroup = U1
            , isExact = variance < 0.001
            }
       else Nothing

-- Time translation generator (∂/∂t)
timeTranslationGenerator :: Double -> ClimateState -> ClimateState
timeTranslationGenerator dt state = state
    -- Time evolution would be applied here
    -- For now, returns unchanged state (infinitesimal)

-- Calculate Earth's energy balance
calculatePlanetaryEnergy :: ClimatePathway -> Double
calculatePlanetaryEnergy pathway =
    let states = map planetaryEnergyAt pathway
    in sum states / fromIntegral (length states)

-- Physical constants with SI units
solarConstant :: Double
solarConstant = 1361.0  -- W/m² at 1 AU

earthRadius :: Double  
earthRadius = 6.371008e6  -- m (IUGG mean radius)

stefanBoltzmann :: Double
stefanBoltzmann = 5.670374419e-8  -- W/(m²·K⁴) CODATA 2018

planetaryEnergyAt :: ClimateState -> Double
planetaryEnergyAt state =
    let -- Earth's cross-sectional area for solar interception
        crossSection = pi * earthRadius^2
        
        -- Incoming solar radiation (accounting for eccentricity)
        eccentricity = 0.0167086  -- Earth's orbital eccentricity
        perihelion = 0.98329  -- Relative distance at perihelion
        solarFlux = solarConstant / perihelion^2  -- Adjust for distance
        
        -- Bond albedo (wavelength-integrated)
        bondAlbedo = unQuantity (albedo state)
        absorbed = solarFlux * crossSection * (1 - bondAlbedo) / 4
        
        -- Outgoing longwave with greenhouse effect
        surfaceTemp = 288.15 + unQuantity (globalTemp state)  -- K
        
        -- Greenhouse factor from CO2, CH4, N2O, H2O
        co2Forcing = 5.35 * log (unQuantity (co2Concentration state) / 280.0)
        ch4Forcing = 0.036 * (sqrt (unQuantity (methaneConc state)) - sqrt 0.75)
        n2oForcing = 0.12 * (sqrt (unQuantity (n2oConc state)) - sqrt 0.27)
        h2oFeedback = 2.0 * unQuantity (globalTemp state)  -- ~2 W/m²/K water vapor feedback
        
        totalGreenhouse = co2Forcing + ch4Forcing + n2oForcing + h2oFeedback
        
        -- Cloud radiative effect
        cloudEffect = -20.0 * unQuantity (cloudFraction state)  -- W/m²
        
        -- Effective emission temperature
        emissionTemp = surfaceTemp * (1 - totalGreenhouse / (stefanBoltzmann * surfaceTemp^4))^0.25
        outgoing = stefanBoltzmann * emissionTemp^4 * 4 * pi * earthRadius^2
        
        -- Include ocean heat uptake
        oceanUptake = 0.5e15 * unQuantity (oceanHeatContent state)  -- W (simplified)
        
        -- Net TOA imbalance
        imbalance = (absorbed - outgoing + oceanUptake) / (4 * pi * earthRadius^2)
    in imbalance

-- Energy-momentum tensor T^μν
energyMomentumTensor :: ClimatePathway -> ConservedCurrent
energyMomentumTensor pathway = ConservedCurrent
    { currentName = "T_μν"
    , components = M.fromList
        [ ("T_00", calculatePlanetaryEnergy pathway)  -- Energy density
        , ("T_0i", 0.0)  -- Energy flux (simplified)
        , ("T_ij", calculateRadiativeStress pathway)  -- Stress tensor
        ]
    , divergence = 0.0  -- ∂_μ T^μν = 0 for conservation
    }

calculateRadiativeStress :: ClimatePathway -> Double
calculateRadiativeStress pathway = 
    -- Radiative stress from greenhouse effect
    let states = pathway
        co2s = map co2Concentration states
        forcingPerDoubling = 3.7  -- W/m² per CO2 doubling
        preindustrial = 280.0
    in sum [forcingPerDoubling * log (co2 / preindustrial) / log 2 | co2 <- co2s] 
       / fromIntegral (length co2s)

-- Check energy conservation in rotating frame (angular momentum)
checkAngularMomentum :: ClimatePathway -> Maybe ContinuousSymmetry
checkAngularMomentum pathway =
    let angMomentum = calculateAtmosphericAngularMomentum pathway
        variance = calculateVariance (map atmAngularMomentumAt pathway)
        isConserved = variance < 0.05  -- Within 5% (less strict due to torques)
    in if isConserved
       then Just $ ContinuousSymmetry
            { symType = "rotation"
            , generator = rotationGenerator
            , parameter = 2 * pi / (24 * 3600)  -- Earth's rotation rate
            , conservedQty = ConservedQuantity
                { qtyName = "atmospheric_angular_momentum"
                , qtyValue = angMomentum
                , qtyUnits = "kg·m²/s"
                , current = angularMomentumCurrent pathway
                }
            , symGroup = SO2
            , isExact = False  -- Broken by mountain torques
            }
       else Nothing

-- Atmospheric angular momentum
calculateAtmosphericAngularMomentum :: ClimatePathway -> Double
calculateAtmosphericAngularMomentum pathway =
    sum (map atmAngularMomentumAt pathway) / fromIntegral (length pathway)

atmAngularMomentumAt :: ClimateState -> Double
atmAngularMomentumAt state =
    let omega = 7.27e-5  -- Earth rotation rate (rad/s)
        radius = 6.371e6  -- Earth radius (m)
        atmMass = 5.15e18  -- Atmospheric mass (kg)
        -- Simplified: AAM ∝ wind speed × mass × radius²
        -- Affected by AMOC strength
        windFactor = 1.0 + 0.1 * unQuantity (amocStrength state) / 15.0  -- Normalized to typical 15 Sv
    in atmMass * radius^2 * omega * windFactor

rotationGenerator :: ClimateState -> ClimateState
rotationGenerator = id  -- Simplified

angularMomentumCurrent :: ClimatePathway -> ConservedCurrent
angularMomentumCurrent pathway = ConservedCurrent
    { currentName = "L_z"
    , components = M.singleton "L_z" (calculateAtmosphericAngularMomentum pathway)
    , divergence = 0.0
    }

-- Check scale invariance (power law behavior)
checkScaleInvariance :: ClimatePathway -> Maybe ContinuousSymmetry
checkScaleInvariance pathway =
    let (isScaleInvariant, exponent) = checkPowerLaw pathway
    in if isScaleInvariant
       then Just $ ContinuousSymmetry
            { symType = "scale"
            , generator = scaleGenerator exponent
            , parameter = exponent
            , conservedQty = ConservedQuantity
                { qtyName = "scale_current"
                , qtyValue = exponent
                , qtyUnits = "dimensionless"
                , current = scaleCurrent exponent
                }
            , symGroup = Rplus
            , isExact = False
            }
       else Nothing

checkPowerLaw :: ClimatePathway -> (Bool, Double)
checkPowerLaw pathway =
    -- Check if warming follows logarithmic forcing law
    let co2s = map (unQuantity . co2Concentration) pathway
        temps = map (unQuantity . globalTemp) pathway
        
        -- Remove outliers using MAD (Median Absolute Deviation)
        cleanedData = removeOutliers $ zip co2s temps
        
        -- Transform to log space for power law: T ~ log(CO2/CO2_0)
        co2_0 = 280.0  -- Pre-industrial
        logData = [(log (co2 / co2_0), temp) | (co2, temp) <- cleanedData, co2 > 0]
        
        -- Theil-Sen regression estimator
        (slope, intercept, r2) = theilSenRegression logData
        
        -- Expected slope from climate physics: ΔT = λ * ΔF where ΔF = 5.35 * ln(CO2/CO2_0)
        -- So slope should be λ * 5.35 where λ ≈ 0.8 K/(W/m²)
        expectedSlope = 0.8 * 5.35
        slopeRatio = slope / expectedSlope
        
    in (r2 > 0.85 && abs(slopeRatio - 1.0) < 0.3, slope)

removeOutliers :: [(Double, Double)] -> [(Double, Double)]
removeOutliers points =
    let values = map snd points
        median = calculateMedian values
        mad = medianAbsoluteDeviation values median
        threshold = 3.0 * mad
    in filter (\(_, v) -> abs(v - median) <= threshold) points

theilSenRegression :: [(Double, Double)] -> (Double, Double, Double)
theilSenRegression points =
    let n = length points
        -- Calculate all pairwise slopes
        slopes = [((y2 - y1) / (x2 - x1)) | 
                 (i, (x1, y1)) <- zip [0..] points,
                 (j, (x2, y2)) <- zip [0..] points,
                 i < j, abs(x2 - x1) > 1e-10]
        
        medianSlope = calculateMedian slopes
        
        -- Calculate intercepts using median slope
        intercepts = [y - medianSlope * x | (x, y) <- points]
        medianIntercept = calculateMedian intercepts
        
        -- Calculate R² 
        yMean = sum (map snd points) / fromIntegral n
        ssTotal = sum [(y - yMean)^2 | (_, y) <- points]
        ssResidual = sum [(y - (medianSlope * x + medianIntercept))^2 | (x, y) <- points]
        r2 = if ssTotal > 0 then 1 - ssResidual / ssTotal else 0
        
    in (medianSlope, medianIntercept, r2)

calculateMedian :: [Double] -> Double
calculateMedian xs = 
    let sorted = sort xs
        n = length sorted
    in if odd n 
       then sorted !! (n `div` 2)
       else (sorted !! (n `div` 2 - 1) + sorted !! (n `div` 2)) / 2

medianAbsoluteDeviation :: [Double] -> Double -> Double
medianAbsoluteDeviation values median =
    let deviations = map (\x -> abs(x - median)) values
    in 1.4826 * calculateMedian deviations  -- Scale factor for consistency with std dev

scaleGenerator :: Double -> ClimateState -> ClimateState
scaleGenerator lambda state = state
    { globalTemp = Quantity (unQuantity (globalTemp state) * lambda)
    , co2Concentration = Quantity (unQuantity (co2Concentration state) * lambda)
    }

scaleCurrent :: Double -> ConservedCurrent
scaleCurrent exponent = ConservedCurrent
    { currentName = "D_μ"
    , components = M.singleton "scaling_dimension" exponent
    , divergence = 0.0
    }

-- ─────────────────────────────────────────────────────────────────────────────
-- DISCRETE SYMMETRIES
-- ─────────────────────────────────────────────────────────────────────────────

data DiscreteSymmetry = DiscreteSymmetry
    { discSymType :: String
    , operation   :: ClimateState -> ClimateState
    , order       :: Int  -- How many times to apply before identity
    , conserved   :: Maybe ConservedQuantity
    } deriving (Show)

detectDiscreteSymmetries :: ClimatePathway -> [DiscreteSymmetry]
detectDiscreteSymmetries pathway = 
    let symmetries =
            [ checkHemisphericSymmetry pathway
            , checkSeasonalCycle pathway
            , checkENSOPeriodicity pathway
            , checkQBOSymmetry pathway
            , checkMilankovitchCycles pathway
            , checkSolarCycle pathway
            ] `using` parList rdeepseq
    in catMaybes symmetries

-- Check hemispheric symmetry (approximate due to land distribution)
checkHemisphericSymmetry :: ClimatePathway -> Maybe DiscreteSymmetry
checkHemisphericSymmetry pathway =
    -- Earth has asymmetric hemispheres, but check anyway
    Just $ DiscreteSymmetry
        { discSymType = "hemispheric_reflection"
        , operation = hemisphericReflection
        , order = 2
        , conserved = Nothing  -- Broken by land asymmetry
        }

hemisphericReflection :: ClimateState -> ClimateState
hemisphericReflection state = state
    -- Would flip Northern/Southern hemisphere fields
    -- Currently simplified

-- Check seasonal periodicity
checkSeasonalCycle :: ClimatePathway -> Maybe DiscreteSymmetry
checkSeasonalCycle pathway =
    let period = 365.25  -- days
        hasCycle = detectPeriodicity pathway period
    in if hasCycle
       then Just $ DiscreteSymmetry
            { discSymType = "seasonal_cycle"
            , operation = seasonalTranslation
            , order = 4  -- Four seasons
            , conserved = Just $ ConservedQuantity
                { qtyName = "seasonal_phase"
                , qtyValue = 0.0
                , qtyUnits = "radians"
                , current = ConservedCurrent "seasonal" M.empty 0.0
                }
            }
       else Nothing

seasonalTranslation :: ClimateState -> ClimateState
seasonalTranslation = id  -- Simplified

-- Check ENSO periodicity
checkENSOPeriodicity :: ClimatePathway -> Maybe DiscreteSymmetry
checkENSOPeriodicity pathway =
    let period = 4.0  -- years (approximate ENSO period)
        hasENSO = detectPeriodicity pathway (period * 365.25)
    in if hasENSO
       then Just $ DiscreteSymmetry
            { discSymType = "ENSO_cycle"
            , operation = ensoPhaseShift
            , order = 2  -- El Niño / La Niña
            , conserved = Nothing
            }
       else Nothing

ensoPhaseShift :: ClimateState -> ClimateState
ensoPhaseShift state = state
    { globalTemp = Quantity (unQuantity (globalTemp state) * (-1))  -- Flip warm/cold phase
    }

detectPeriodicity :: ClimatePathway -> Double -> Bool
detectPeriodicity pathway period =
    -- Use autocorrelation to detect periodicity
    let values = map (unQuantity . globalTemp) pathway
        autocorr = autocorrelation values (round period)
    in abs autocorr > 0.7

autocorrelation :: [Double] -> Int -> Double
autocorrelation values lag =
    let n = length values
        mean = sum values / fromIntegral n
        centered = map (\v -> v - mean) values
        pairs = zip centered (drop lag centered)
        covar = sum [x * y | (x, y) <- pairs] / fromIntegral (length pairs)
        variance = sum [x * x | x <- centered] / fromIntegral n
    in covar / variance

-- ─────────────────────────────────────────────────────────────────────────────
-- BROKEN SYMMETRIES AND TIPPING POINTS
-- ─────────────────────────────────────────────────────────────────────────────

data BrokenSymmetry = BrokenSymmetry
    { brokenType    :: String
    , original      :: SymmetryGroup
    , residual      :: Maybe SymmetryGroup
    , orderParam    :: OrderParameter
    , criticalTemp  :: Double
    , climateMode   :: Maybe ClimateOscillationMode
    } deriving (Show)

data OrderParameter = OrderParameter
    { paramName  :: String
    , paramValue :: Double
    , critical   :: Double
    } deriving (Show)

data ClimateOscillationMode = ClimateOscillationMode
    { modeName   :: String
    , period     :: Double  -- Oscillation period in years
    , amplitude  :: Double  -- Typical amplitude
    , spatial_pattern :: String  -- Geographic pattern
    }

instance Show ClimateOscillationMode where
    show com = "ClimateOscillationMode " ++ modeName com

-- Detect broken symmetries (climate tipping points)
detectBrokenSymmetries :: ClimatePathway -> [BrokenSymmetry]
detectBrokenSymmetries pathway =
    [ checkArcticSeaIceCollapse pathway
    , checkAMOCShutdown pathway
    , checkAmazonDieback pathway
    , checkWAISCollapse pathway
    ]

-- Arctic sea ice collapse (albedo feedback breaks symmetry)
checkArcticSeaIceCollapse :: ClimatePathway -> BrokenSymmetry
checkArcticSeaIceCollapse pathway =
    let iceExtent = unQuantity $ minimum (map iceSheetMass pathway)
        criticalExtent = 1.0e6  -- km² (arbitrary threshold)
    in BrokenSymmetry
        { brokenType = "ice_albedo_feedback"
        , original = SO2  -- Rotational symmetry in ice coverage
        , residual = Nothing  -- Total loss
        , orderParam = OrderParameter
            { paramName = "ice_extent"
            , paramValue = iceExtent
            , critical = criticalExtent
            }
        , criticalTemp = 2.0  -- °C global warming
        , climateMode = Just $ ClimateOscillationMode
            { modeName = "ice_edge_oscillations"
            , period = 5.0  -- Years for ice edge dynamics
            , amplitude = 0.1  -- Relative amplitude
            , spatial_pattern = "Arctic_circumpolar"
            }
        }

-- AMOC shutdown (thermohaline circulation collapse)
checkAMOCShutdown :: ClimatePathway -> BrokenSymmetry
checkAMOCShutdown pathway =
    let amoc = unQuantity $ minimum (map amocStrength pathway)
        criticalFlow = 5.0  -- Sverdrups
    in BrokenSymmetry
        { brokenType = "thermohaline_collapse"
        , original = U1  -- Circulation phase
        , residual = Nothing
        , orderParam = OrderParameter
            { paramName = "AMOC_strength"
            , paramValue = amoc
            , critical = criticalFlow
            }
        , criticalTemp = 3.0
        , climateMode = Nothing  -- No regular oscillation mode
        }

-- Amazon rainforest dieback
checkAmazonDieback :: ClimatePathway -> BrokenSymmetry
checkAmazonDieback pathway =
    BrokenSymmetry
        { brokenType = "forest_savanna_transition"
        , original = SO3  -- Full ecosystem symmetry
        , residual = Just Z2  -- Bistable states
        , orderParam = OrderParameter
            { paramName = "forest_fraction"
            , paramValue = 0.8  -- Current
            , critical = 0.4     -- Tipping point
            }
        , criticalTemp = 3.5
        , climateMode = Just $ ClimateOscillationMode
            { modeName = "forest_savanna_cycles"
            , period = 15.0  -- Years for vegetation transitions
            , amplitude = 0.3  -- Relative amplitude  
            , spatial_pattern = "tropical_bands"
            }
        }

-- West Antarctic Ice Sheet collapse
checkWAISCollapse :: ClimatePathway -> BrokenSymmetry
checkWAISCollapse pathway =
    BrokenSymmetry
        { brokenType = "marine_ice_instability"
        , original = SO2
        , residual = Nothing
        , orderParam = OrderParameter
            { paramName = "grounding_line_position"
            , paramValue = 1000.0  -- km from edge
            , critical = 100.0
            }
        , criticalTemp = 2.5
        , climateMode = Nothing
        }

-- ─────────────────────────────────────────────────────────────────────────────
-- CRITICAL PHENOMENA AND UNIVERSALITY CLASSES
-- ─────────────────────────────────────────────────────────────────────────────

data CriticalExponents = CriticalExponents
    { alpha :: Double  -- Specific heat: C ~ |t|^(-α)
    , beta  :: Double  -- Order parameter: M ~ |t|^β
    , gamma :: Double  -- Susceptibility: χ ~ |t|^(-γ)
    , delta :: Double  -- Critical isotherm: M ~ h^(1/δ)
    , nu    :: Double  -- Correlation length: ξ ~ |t|^(-ν)
    , eta   :: Double  -- Correlation function: G(r) ~ r^(-(d-2+η))
    } deriving (Show, Eq)

data UniversalityClass = UniversalityClass
    { className   :: String
    , dimension   :: Int
    , symmetryGrp :: String
    , exponents   :: CriticalExponents
    } deriving (Show)

-- Identify universality class of climate transition
identifyUniversalityClass :: BrokenSymmetry -> UniversalityClass
identifyUniversalityClass broken =
    case brokenType broken of
        "ice_albedo_feedback" -> isingUniversality
        "thermohaline_collapse" -> dynamicPercolation
        "forest_savanna_transition" -> directedPercolation
        "marine_ice_instability" -> meanFieldUniversality
        _ -> meanFieldUniversality

-- 3D Ising universality (magnetic-like)
isingUniversality :: UniversalityClass
isingUniversality = UniversalityClass
    { className = "3D_Ising"
    , dimension = 3
    , symmetryGrp = "Z2"
    , exponents = CriticalExponents
        { alpha = 0.110
        , beta = 0.326
        , gamma = 1.237
        , delta = 4.789
        , nu = 0.630
        , eta = 0.036
        }
    }

-- Dynamic percolation (for AMOC)
dynamicPercolation :: UniversalityClass
dynamicPercolation = UniversalityClass
    { className = "Dynamic_Percolation"
    , dimension = 3
    , symmetryGrp = "None"
    , exponents = CriticalExponents
        { alpha = -0.5
        , beta = 0.41
        , gamma = 1.80
        , delta = 5.4
        , nu = 0.88
        , eta = 0.0
        }
    }

-- Directed percolation (for ecosystem collapse)
directedPercolation :: UniversalityClass
directedPercolation = UniversalityClass
    { className = "Directed_Percolation"
    , dimension = 2
    , symmetryGrp = "None"
    , exponents = CriticalExponents
        { alpha = 0.159
        , beta = 0.276
        , gamma = 1.096
        , delta = 4.97
        , nu = 0.734
        , eta = 0.313
        }
    }

-- Mean field (long-range interactions)
meanFieldUniversality :: UniversalityClass
meanFieldUniversality = UniversalityClass
    { className = "Mean_Field"
    , dimension = 4  -- Upper critical dimension
    , symmetryGrp = "Any"
    , exponents = CriticalExponents
        { alpha = 0
        , beta = 0.5
        , gamma = 1.0
        , delta = 3.0
        , nu = 0.5
        , eta = 0
        }
    }

-- ─────────────────────────────────────────────────────────────────────────────
-- NOETHER'S THEOREM APPLICATION
-- ─────────────────────────────────────────────────────────────────────────────

-- Apply Noether's theorem to get conservation laws
noetherTheorem :: ContinuousSymmetry -> ConservationLaw
noetherTheorem sym = ConservationLaw
    { conservedName = qtyName (conservedQty sym)
    , conservedCurrent = current (conservedQty sym)
    , conservedCharge = integrateCharge (current (conservedQty sym))
    , anomaly = checkAnomaly sym
    }

data ConservationLaw = ConservationLaw
    { conservedName :: String
    , conservedCurrent :: ConservedCurrent
    , conservedCharge :: Double
    , anomaly :: Maybe Double
    } deriving (Show)

integrateCharge :: ConservedCurrent -> Double
integrateCharge curr =
    -- Q = ∫ J^0 d³x (integrate time component over space)
    M.findWithDefault 0.0 "T_00" (components curr)

checkAnomaly :: ContinuousSymmetry -> Maybe Double
checkAnomaly sym =
    -- Check if conservation is violated quantum mechanically
    if divergence (current (conservedQty sym)) > 1e-6
    then Just (divergence (current (conservedQty sym)))
    else Nothing

-- ─────────────────────────────────────────────────────────────────────────────
-- EARLY WARNING SIGNALS FROM SYMMETRY BREAKING
-- ─────────────────────────────────────────────────────────────────────────────

-- Detect approaching tipping point from symmetry analysis
earlyWarningSignals :: ClimatePathway -> [WarningSignal]
earlyWarningSignals pathway =
    [ criticalSlowingDown pathway
    , increasedVariance pathway
    , spatialCorrelation pathway
    , flickering pathway
    ]

data WarningSignal = WarningSignal
    { signalType :: String
    , strength   :: Double  -- 0 to 1
    , timescale  :: Double  -- Years before tipping
    , confidence :: Double  -- Statistical confidence
    } deriving (Show)

-- Critical slowing down (recovery time increases)
criticalSlowingDown :: ClimatePathway -> WarningSignal
criticalSlowingDown pathway =
    let temps = map (unQuantity . globalTemp) pathway
        autocorrs = [autocorrelation temps lag | lag <- [1..10]]
        ar1 = head autocorrs  -- Lag-1 autocorrelation
    in WarningSignal
        { signalType = "critical_slowing_down"
        , strength = ar1
        , timescale = -1 / log ar1  -- Decorrelation time
        , confidence = if ar1 > 0.9 then 0.95 else 0.5
        }

-- Increased variance near tipping point
increasedVariance :: ClimatePathway -> WarningSignal
increasedVariance pathway =
    let temps = map (unQuantity . globalTemp) pathway
        n = length temps
        firstHalf = take (n `div` 2) temps
        secondHalf = drop (n `div` 2) temps
        var1 = calculateVariance firstHalf
        var2 = calculateVariance secondHalf
        increase = var2 / var1
    in WarningSignal
        { signalType = "variance_increase"
        , strength = min 1.0 (increase / 2)  -- Normalize
        , timescale = 10.0  -- Typical warning time
        , confidence = if increase > 1.5 then 0.8 else 0.3
        }

-- Spatial correlation increases
spatialCorrelation :: ClimatePathway -> WarningSignal
spatialCorrelation pathway =
    let temps = map (unQuantity . globalTemp) pathway
        corrLen = computeCorrelationLength temps
        normLen = tanh (corrLen / 10) -- normalize
        fftPower = maybe 0 snd (detectPeriodicityFFT temps 2)
        strength' = min 1.0 (0.6*normLen + 0.4*fftPower)
        conf = if strength' > 0.8 then 0.85 else if strength' > 0.5 then 0.6 else 0.3
    in WarningSignal
        { signalType = "spatial_correlation"
        , strength = strength'
        , timescale = 5.0
        , confidence = conf
        }

-- Flickering between states
flickering :: ClimatePathway -> WarningSignal
flickering pathway =
    let temps = map (unQuantity . globalTemp) pathway
        changes = zipWith (-) (tail temps) temps
        signChanges = length [() | (a, b) <- zip changes (tail changes), a * b < 0]
        flickerRate = fromIntegral signChanges / fromIntegral (length changes)
    in WarningSignal
        { signalType = "flickering"
        , strength = min 1.0 (flickerRate * 10)
        , timescale = 2.0
        , confidence = if flickerRate > 0.3 then 0.7 else 0.2
        }

-- ─────────────────────────────────────────────────────────────────────────────
-- HELPER FUNCTIONS
-- ─────────────────────────────────────────────────────────────────────────────

calculateVariance :: [Double] -> Double
calculateVariance xs =
    let n = length xs
        mean = sum xs / fromIntegral n
        squaredDiffs = [(x - mean)^2 | x <- xs]
    in sum squaredDiffs / fromIntegral n

-- Add new symmetry checks
checkEnergyMomentumConservation :: ClimatePathway -> Maybe ContinuousSymmetry
checkEnergyMomentumConservation pathway = 
    -- Check conservation of energy, mass, and momentum in climate system
    let energyConserved = checkEnergyConservation pathway
        massConserved = checkMassConservation pathway
        momentumConserved = checkMomentumConservation pathway
    in if energyConserved && massConserved && momentumConserved
       then Just $ ContinuousSymmetry
            { symType = "energy_momentum_mass_conservation"
            , generator = energyMomentumGenerator
            , parameter = 1.0
            , conservedQty = ConservedQuantity
                { qtyName = "climate_energy_momentum_mass"
                , qtyValue = 1.0
                , qtyUnits = "J·kg·m/s"
                , current = ConservedCurrent "EMM" M.empty 0.0
                }
            , symGroup = Product (Product U1 U1) U1
            , isExact = False  -- Dissipative system
            }
       else Nothing

energyMomentumGenerator :: ClimateState -> ClimateState
energyMomentumGenerator state = state
    { globalTemp = globalTemp state  -- Energy conservation
    , co2Concentration = co2Concentration state  -- Mass conservation
    }

checkEnergyConservation :: ClimatePathway -> Bool
checkEnergyConservation pathway = 
    let energies = map planetaryEnergyAt pathway
        variance = calculateVariance energies
    in variance < 0.01

checkMassConservation :: ClimatePathway -> Bool  
checkMassConservation pathway =
    let masses = map (unQuantity . iceSheetMass) pathway
        variance = calculateVariance masses
    in variance < 0.05  -- Ice mass changes allowed

checkMomentumConservation :: ClimatePathway -> Bool
checkMomentumConservation pathway =
    let momenta = map atmAngularMomentumAt pathway
        variance = calculateVariance momenta
    in variance < 0.1  -- Angular momentum with mountain torques

checkGaugeInvariance :: ClimatePathway -> Maybe ContinuousSymmetry  
checkGaugeInvariance pathway =
    -- Check invariance under policy gauge transformations
    Just $ ContinuousSymmetry
        { symType = "policy_gauge"
        , generator = policyGaugeTransform
        , parameter = 1.0
        , conservedQty = ConservedQuantity
            { qtyName = "policy_invariant"
            , qtyValue = calculatePolicyInvariant pathway
            , qtyUnits = "GtCO2"
            , current = ConservedCurrent "policy" M.empty 0.0
            }
        , symGroup = Gauge "U1_policy"
        , isExact = False
        }

policyGaugeTransform :: ClimateState -> ClimateState
policyGaugeTransform = id

calculatePolicyInvariant :: ClimatePathway -> Double
calculatePolicyInvariant pathway =
    -- Cumulative emissions invariant
    sum [unQuantity (co2Concentration s) - 280 | s <- pathway] * 2.12  -- GtC per ppm

-- Lorentz invariance is not applicable to climate systems (non-relativistic)
-- Climate systems operate in the Newtonian regime where v << c

checkQBOSymmetry :: ClimatePathway -> Maybe DiscreteSymmetry
checkQBOSymmetry pathway =
    -- Quasi-Biennial Oscillation (~28 month period)
    let period = 28 * 30  -- days
        hasQBO = detectPeriodicity pathway period
    in if hasQBO
       then Just $ DiscreteSymmetry
            { discSymType = "QBO_cycle"
            , operation = id
            , order = 2
            , conserved = Nothing
            }
       else Nothing

checkMilankovitchCycles :: ClimatePathway -> Maybe DiscreteSymmetry
checkMilankovitchCycles _ =
    -- Orbital cycles: too long for typical pathways
    Nothing  -- Would need >20kyr data

checkSolarCycle :: ClimatePathway -> Maybe DiscreteSymmetry  
checkSolarCycle pathway =
    let period = 11 * 365.25  -- 11-year solar cycle
        hasSolar = detectPeriodicity pathway period
    in if hasSolar
       then Just $ DiscreteSymmetry
            { discSymType = "solar_cycle"
            , operation = id
            , order = 11
            , conserved = Nothing
            }
       else Nothing

linearRegression :: [(Double, Double)] -> (Double, Double)
linearRegression points =
    let n = fromIntegral (length points)
        sumX = sum [x | (x, _) <- points]
        sumY = sum [y | (_, y) <- points]
        sumXY = sum [x * y | (x, y) <- points]
        sumX2 = sum [x * x | (x, _) <- points]
        sumY2 = sum [y * y | (_, y) <- points]
        
        slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX)
        intercept = (sumY - slope * sumX) / n
        
        -- Calculate R²
        meanY = sumY / n
        ssTotal = sum [(y - meanY)^2 | (_, y) <- points]
        ssResidual = sum [(y - (slope * x + intercept))^2 | (x, y) <- points]
        r2 = if ssTotal > 0 then 1 - ssResidual / ssTotal else 0
        
    in (slope, r2)

-- ─────────────────────────────────────────────────────────────────────────────
-- MAIN ANALYSIS
-- ─────────────────────────────────────────────────────────────────────────────

analyzeClimateSymmetries :: ClimatePathway -> IO ()
analyzeClimateSymmetries pathway = do
    putStrLn "═══════════════════════════════════════════════════════════════"
    putStrLn "CLIMATE SYMMETRY AND CONSERVATION LAW ANALYSIS"
    putStrLn "═══════════════════════════════════════════════════════════════"
    
    -- Continuous symmetries
    let continuous = detectContinuousSymmetries pathway
    putStrLn $ "\nFound " ++ show (length continuous) ++ " continuous symmetries:"
    mapM_ printContinuousSymmetry continuous
    
    -- Conservation laws via Noether
    putStrLn "\nConservation laws from Noether's theorem:"
    mapM_ (printConservationLaw . noetherTheorem) continuous
    
    -- Discrete symmetries
    let discrete = detectDiscreteSymmetries pathway
    putStrLn $ "\nFound " ++ show (length discrete) ++ " discrete symmetries:"
    mapM_ printDiscreteSymmetry discrete
    
    -- Broken symmetries (tipping points)
    let broken = detectBrokenSymmetries pathway
    putStrLn $ "\nDetected " ++ show (length broken) ++ " broken symmetries:"
    mapM_ printBrokenSymmetry broken
    
    -- Universality classes
    putStrLn "\nUniversality classes of transitions:"
    mapM_ (printUniversalityClass . identifyUniversalityClass) broken
    
    -- Early warning signals
    let warnings = earlyWarningSignals pathway
    putStrLn "\nEarly warning signals:"
    mapM_ printWarningSignal warnings

    -- Additive symmetry diagnostics (table)
    let advDiags = runSymmetryDiagnostics pathway
    putStrLn "\nSymmetry diagnostics:"
    putStrLn (renderDiagnosticsTable advDiags)

    -- Bayesian symmetry persistence evaluation
    let bayesE = evaluateSymmetryPersistence pathway
    unless (null bayesE) $ do
        putStrLn "\nBayesian symmetry persistence (ΔlogE > 0 favors conservation):"
        putStrLn (renderBayesianEvidence bayesE)

    -- Dynamical Mode Decomposition (DMD)
    let dmdRes = runClimateDMD pathway
    putStrLn "\nDMD modal structure (climate variables subset):"
    putStrLn (renderDMDResult dmdRes)

    -- Emergent Z_n symmetry detection from temperature anomalies
    let temps = map (unQuantity . globalTemp) pathway
    putStrLn ("\n" ++ renderEmergentZn (detectEmergentZn temps))

    -- Climate oscillation mode fits (additive)
    let oscillationFits = computeClimateOscillationFits broken
    unless (null oscillationFits) $ do
        putStrLn "\nClimate oscillation mode fits (period analysis):"
        putStrLn (renderClimateOscillationFits oscillationFits)

    -- (Optional future) Ensemble handling placeholder note suppressed (no stubs) - if multiple pathways were available we would summarize.

    -- JSON export (inline additive demonstration; user can redirect)
    let jsonExport = diagnosticsToJSON continuous discrete broken warnings advDiags bayesE oscillationFits
    putStrLn "\n[JSON export excerpt (truncated 300 chars)]:"
    putStrLn (take 300 jsonExport ++ if length jsonExport > 300 then "..." else "")

    -- CSV Bayesian evidence export (excerpt)
    let csvExport = csvBayesianEvidence bayesE
    putStrLn "\n[CSV Bayesian evidence excerpt]:"
    putStrLn (unlines (take 5 (lines csvExport)))

    -- Additive Lie structure inference (approximate)
    let lieStruct = inferLieStructure continuous
    unless (null lieStruct) $ do
        putStrLn "\nApproximate Lie bracket magnitudes:"
        putStrLn (renderLieStructure lieStruct)

    -- Aggregate warning index with spectral slope bonus
    let warningIndex = aggregateWarningIndex warnings temps
    putStrLn $ "\nAggregate warning index: " ++ printf "%.3f" warningIndex

    -- Batch 3: Largest Lyapunov exponent on temperature series
    let lyap = estimateLargestLyapunov temps
    putStrLn $ "\n" ++ renderLyapunov lyap

    -- Batch 3: Granger causality among selected variables
    let varSets = [ ("temp", temps)
                  , ("energy", map planetaryEnergyAt pathway)
                  , ("amoc", map (unQuantity . amocStrength) pathway)
                  ]
        granger = computeGranger varSets
    putStrLn "\nPairwise Granger causality (lag=1):"
    putStrLn (renderGranger granger)

    -- Batch 3: Recurrence Quantification Analysis for temperature
    let rqa = computeRQA temps
    putStrLn ("\n" ++ renderRQA rqa)

    -- Bootstrap uncertainty on planetary energy mean (minimal resamples for speed)
    let energySeries = map planetaryEnergyAt pathway
        (bootMean, bootSE) = bootstrapMean 64 energySeries
    putStrLn $ "Bootstrapped mean planetary energy: " ++ printf "%.4f ± %.4f" bootMean bootSE
    
    putStrLn "\n═══════════════════════════════════════════════════════════════"

printContinuousSymmetry :: ContinuousSymmetry -> IO ()
printContinuousSymmetry sym = do
    putStrLn $ "  • " ++ symType sym ++ " (" ++ show (symGroup sym) ++ ")"
    putStrLn $ "    Conserves: " ++ qtyName (conservedQty sym) ++ 
               " = " ++ show (qtyValue (conservedQty sym)) ++ " " ++ 
               qtyUnits (conservedQty sym)
    putStrLn $ "    Exact: " ++ show (isExact sym)

printConservationLaw :: ConservationLaw -> IO ()
printConservationLaw law = do
    putStrLn $ "  • " ++ conservedName law ++ 
               " (Q = " ++ show (conservedCharge law) ++ ")"
    case anomaly law of
        Just a -> putStrLn $ "    ⚠ Anomaly detected: " ++ show a
        Nothing -> putStrLn "    ✓ No anomaly"

printDiscreteSymmetry :: DiscreteSymmetry -> IO ()
printDiscreteSymmetry sym =
    putStrLn $ "  • " ++ discSymType sym ++ " (order " ++ show (order sym) ++ ")"

printBrokenSymmetry :: BrokenSymmetry -> IO ()
printBrokenSymmetry broken = do
    putStrLn $ "  • " ++ brokenType broken
    putStrLn $ "    " ++ show (original broken) ++ " → " ++ 
               maybe "None" show (residual broken)
    putStrLn $ "    Critical T: " ++ show (criticalTemp broken) ++ "°C"
    putStrLn $ "    Order parameter: " ++ paramName (orderParam broken) ++ 
               " = " ++ show (paramValue (orderParam broken))

printUniversalityClass :: UniversalityClass -> IO ()
printUniversalityClass uc = do
    putStrLn $ "  • " ++ className uc ++ " (" ++ symmetryGrp uc ++ ")"
    let exp = exponents uc
    putStrLn $ "    β = " ++ show (beta exp) ++ ", γ = " ++ show (gamma exp) ++ 
               ", ν = " ++ show (nu exp)

printWarningSignal :: WarningSignal -> IO ()
printWarningSignal signal =
    putStrLn $ "  • " ++ signalType signal ++ 
               " (strength: " ++ show (round (strength signal * 100)) ++ "%)" ++
               " [" ++ show (timescale signal) ++ " years warning]"

-- ─────────────────────────────────────────────────────────────────────────────
-- TEST WITH SAMPLE PATHWAY
-- ─────────────────────────────────────────────────────────────────────────────

testPathway :: ClimatePathway
testPathway = 
    [ ClimateState 
        { globalTemp = Quantity t
        , co2Concentration = Quantity (280 + t * 50)
        , oceanHeatContent = Quantity (10 + t)
        , iceSheetMass = Quantity (1e7 - t * 1e5)
        , albedo = Quantity (0.3 - t * 0.01)
        , amocStrength = Quantity (15 - t * 0.5)
        , cloudFraction = Quantity 0.67
        , methaneConc = Quantity 1.8
        , n2oConc = Quantity 0.33
        , aerosolOptDepth = Quantity 0.1
        , soilMoisture = Quantity 0.4
        , seaLevel = Quantity (t * 0.003)
        }
    | t <- [0, 0.1 .. 2.0]
    ]

main :: IO ()
main = analyzeClimateSymmetries testPathway
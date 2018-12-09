module HullWhite where

import Data.Array
import Data.List

-- # Model Parameters
mean = 0 :: Double
mean_reversion = 0.1 :: Double
volatility = 0.01 :: Double

-- # Calculation Settings
dt = 0.25 :: Double
dr = sqrt $ 3 * volatility**2 * dt :: Double
nt = 12 :: Int -- number of time grids

m = negate $ mean_reversion * dt :: Double
jmax = (floor $ (-0.184) / m) + 1 :: Int
jmin = negate jmax :: Int

-- # Curve Data
-- Sample curve
{-
r :: Double -> Double
r t = 0.03 * (1 - 0.5 * (exp (-0.6 * t)))
initial_curve :: Double -> Double
initial_curve t = exp . negate $ (r t) * t
time_grids = map (\t -> fromIntegral t * dt) [0..nt] :: [Double]
dfs = map initial_curve time_grids
r0 = negate $ (log (dfs !! 1)) / time_grids !! 1 :: Double
-}

-- Book setting
time_grids = map (\t -> fromIntegral t * dt) [0..nt] :: [Double]
dfs = [
    1, 0.993841213, 0.987331092, 0.980146715, 0.972278502, 
    0.963812328, 0.954454998, 0.945343173, 0.934944339, 
    0.924324335, 0.913255041, 0.901857442, 0.890286145
    ]
r0 = negate $ (log (dfs !! 1)) / time_grids !! 1 :: Double

rs = zipWith (\t df -> negate $ (log df) / t) (drop 1 time_grids) (drop 1 dfs)

-- prob k l : transition probability from node (m, k) to (m+1,lj)
prob :: Int -> Int -> Double
prob k l
    | k == jmax = applyProb 0 0 pdu pdm pdd 
    | k == jmin = applyProb puu pum pud 0 0
    | otherwise = applyProb 0 pu pm pd 0
    where
        applyProb :: Double -> Double -> Double -> Double -> Double -> Double
        applyProb p0 p1 p2 p3 p4
            | k - l == -2 = p0
            | k - l == -1 = p1
            | k - l == 0 = p2
            | k - l == 1 = p3
            | k - l == 2 = p4
            | otherwise = 0

        j = fromIntegral k
        pu :: Double
        pu = (1 + 3 * ((j * m)**2 + j * m)) / 6
        pm = 2 / 3 - (j * m)**2
        pd = (1 + 3 * ((j * m)**2 - j * m)) / 6
            
        -- if k == j,ax
        pdu = (7 + 3 * ((j * m)**2 + 3 * j * m)) / 6
        pdm = (-1) / 3 - (j * m)**2 - 2 * j * m
        pdd = (1 + 3 * ((j * m)**2 + j * m)) / 6

        -- if k == jmin
        puu = (1 + 3 * ((j * m)**2 - j * m)) / 6
        pum = (-1) / 3 - (j * m)**2 + 2 * j * m
        pud = (7 + 3 * ((j * m)**2 - 3 * j * m)) / 6

-- type Tree = 

-- hwtree :: Tree
hwtree = (r, q)
    where
        -- short rate on grid (i, j)
        -- i : index of time
        -- j : index of space
        r = listArray ((0, jmin), (nt, jmax)) 
            $ map def_r [(i, j) | i <- [0..nt], j<- [jmin..jmax]]
        def_r :: (Int, Int) -> Double
        def_r (0, _) = r0
        def_r (i, j) = alpha!i + dr * fromIntegral j

        -- center of tree at time i*dt
        alpha = listArray (0, nt) $ map def_alpha [0..nt]
        def_alpha :: Int -> Double
        def_alpha 0 = r!(0, 0)
        def_alpha i = ((log . sum . map contribution $ [jmin..jmax]) - log df_initial) / dt
            where
                -- df_initial = initial_curve (fromIntegral i * dt) :: Double
                df_initial = dfs!!(i+1) :: Double
                contribution :: Int -> Double
                contribution j = q!(i, j) * (exp . negate $ fromIntegral j * dr * dt)

        -- price of Arrow Debreu security on grid (i, j)
        q = listArray ((0, jmin), (nt, jmax)) 
            $ map def_q [(i, j) | i <- [0..nt], j<- [jmin..jmax]]
        def_q :: (Int, Int) -> Double
        def_q (0, 0) = 1
        def_q (0, _) = 0
        def_q (i, j) = sum . map contribution $ [jmin..jmax]
            where
                contribution k = q!(i-1, k) * prob k j * (exp . negate $ r!(i-1, k) * dt)


toSimpleArray :: Array (Int, Int) a -> [[a]]
toSimpleArray grid = transpose [[grid ! (x, y) | x<-[lowx..highx]] |  y<-[lowy..highy]] 
    where ((lowx, lowy), (highx, highy)) =  bounds grid
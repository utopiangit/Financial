import HullWhite 

import Data.List
import Data.Array

type Tree = Array (Int, Int) Double
type Model = (Tree, Tree)

-- libor :: Double -> Double -> Tree -> Double
-- libor :: (Int, Int) -> Int -> Tree -> Double
-- libor (i, j) di tree = 

{-
-- backward :: (Int, Int) -> Tree -> Double
backward (i, j) (r, q) = sum . map contribution $ [jmin..jmax]
    where
        contribution k = prob i k * (exp . negate $ r!(i-1, k) * dt)
-}

backward :: Int -> Int -> (Int -> Double) -> Model -> (Int -> Double)
backward from to payoff (r, q) = \j -> vtree!(to, j)
    where
        vtree :: Array (Int, Int) Double
        vtree = listArray ((to, jmin), (from, jmax)) 
            $ map defv $ [(i, j) | i <- [to..from], j<- [jmin..jmax]]

        defv (i, j) -- i == fromのときのペイオフをどう表現しよう
            | i == from = payoff j -- 関数
            -- | i == from = payoff!!(j - jmin) -- 配列
            | otherwise = sum . map contribution $ [jmin..jmax]
                where 
                    contribution k = vtree!(i+1, k) 
                        * prob j k * (exp . negate $ r!(i, k) * dt)

libor :: (Int, Int) -> Int -> Model -> Double
libor (i, j) tenor model = (1 / df_end - 1) / (dt * fromIntegral tenor)
    where 
        df_end = (backward (i + tenor) i (\_ -> 1) model) j
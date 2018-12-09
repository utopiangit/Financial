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

backward :: Int -> Int -> Array Int Double -> Model -> [Double]
backward from to payoff (r, q) = map (\j -> vtree!(to, j)) [jmin..jmax]
    where
        vtree :: Array (Int, Int) Double
        vtree = listArray ((0, jmin), (nt, jmax)) 
            $ map defv $ [(i, j) | i <- [to..from], j<- [jmin..jmax]]

        defv (i, j) 
            | i == from = payoff!j
            | otherwise = sum . map contribution $ [jmin..jmax]
                where 
                    contribution k = vtree!(i+1, k) 
                        * prob j (j+1) * (exp . negate $ r!(i, k) * dt)

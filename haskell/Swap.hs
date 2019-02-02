import HullWhite 

import Data.List
import Data.Array


-- libor :: Double -> Double -> Tree -> Double
-- libor :: (Int, Int) -> Int -> Tree -> Double
-- libor (i, j) di tree = 

{-
-- backward :: (Int, Int) -> Tree -> Double
backward (i, j) (r, q) = sum . map contribution $ [jmin..jmax]
    where
        contribution k = prob i k * (exp . negate $ r!(i-1, k) * dt)
-}

-- Both from and to are time index. from is after to, i.e. from > to
backward :: Int -> Int -> (Int -> Double) -> Model -> (Int -> Double)
backward from to payoff model = \j -> vtree!(to, j)
    where
        (jmin, jmax) = spaceBoundsOn from model
        vtree :: Array (Int, Int) Double
        vtree = listArray ((to, jmin), (from, jmax)) 
            $ map defv $ [(i, j) | i <- [to..from], j<- [jmin..jmax]]

        defv (i, j) -- i == fromのときのペイオフをどう表現しよう
            | i == from = payoff j -- 関数
            -- | i == from = payoff!!(j - jmin) -- 配列
            | otherwise = sum . map contribution $ [jmin..jmax]
                where 
                    (jmin, jmax) = spaceBoundsOn (i+1) model
                    contribution k = vtree!(i+1, k) 
                        * (transitionProb (i, j) (i+1, k) model) 
                        * (exp . negate $ (rOn (i, k) model) * timeStep i model)


-- ideal signature                        
-- libor :: Double -> Double -> model -> Double
-- libor :: t_fixing -> tenor -> model -> LIBOR
libor :: Int -> Int -> Model -> Double
libor i_fixing i_tenor model = average [ liborOn (i_fixing, j) i_tenor model | j <- [jmin..jmax]]
    where
        (jmin, jmax) = spaceBoundsOn i_fixing model
-- 
-- libor :: (time_index, space_index) -> tenor -> model -> LIBOR
liborOn :: (Int, Int) -> Int -> Model -> Double
liborOn (i, j) tenor model = (1 / df_end - 1) / ((timeStep i model) * fromIntegral tenor)
    where 
        df_end = (backward (i + tenor) i (\_ -> 1) model) j

average :: (Real a) => [a] -> Double
average xs = (realToFrac $ sum xs) / fromIntegral (length xs)
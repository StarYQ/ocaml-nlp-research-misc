open Perceptron

let sigmoid x = 1.0 /. (1.0 +. exp (~-.x));;
let relu x = max 0.0 x;;
let step x = if x>=0.0 then 1.0 else 0.0;;

let inputs = [|1.0; 3.0|];;
let weights = [|2.0; 4.5|];;
let p1 = Perceptron.create weights ~-.14.45 sigmoid;;
let p2 = Perceptron.create weights 0.63 relu;;
let p3 = Perceptron.create weights ~-.0.32 step;;


let () =
  let res1 = Perceptron.output p1 inputs in
  let res2 = Perceptron.output p2 inputs in 
  let res3 = Perceptron.output p3 inputs in
  Printf.printf "Sigmoid: %f\n ReLU: %f\n Heaviside Step: %f\n" 
  res1 res2 res3;;
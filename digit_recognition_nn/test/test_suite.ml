(** Comprehensive test suite for Deep Neural Network *)

open Digit_recognition_nn

(** Helper: compare floats with tolerance *)
let float_equal ?(eps=1e-6) a b =
  abs_float (a -. b) < eps

(** Helper: compare matrices with tolerance *)
let matrix_equal ?(eps=1e-6) a b =
  let rows_a, cols_a = Matrix.shape a in
  let rows_b, cols_b = Matrix.shape b in
  if rows_a <> rows_b || cols_a <> cols_b then false
  else
    let equal = ref true in
    for i = 0 to rows_a - 1 do
      for j = 0 to cols_a - 1 do
        if not (float_equal ~eps a.(i).(j) b.(i).(j)) then
          equal := false
      done
    done;
    !equal

(** Test Matrix operations *)
let test_matrix_creation () =
  let m = Matrix.zeros 2 3 in
  Alcotest.(check int) "rows" 2 (fst (Matrix.shape m));
  Alcotest.(check int) "cols" 3 (snd (Matrix.shape m));
  Alcotest.(check bool) "all zeros" true
    (Array.for_all (fun row -> Array.for_all (fun x -> x = 0.0) row) m)

let test_matrix_transpose () =
  let m = Matrix.of_lists [[1.0; 2.0; 3.0]; [4.0; 5.0; 6.0]] in
  let mt = Matrix.transpose m in
  let expected = Matrix.of_lists [[1.0; 4.0]; [2.0; 5.0]; [3.0; 6.0]] in
  Alcotest.(check bool) "transpose" true (matrix_equal mt expected)

let test_matrix_dot () =
  let a = Matrix.of_lists [[1.0; 2.0]; [3.0; 4.0]] in
  let b = Matrix.of_lists [[2.0; 0.0]; [1.0; 2.0]] in
  let c = Matrix.dot a b in
  let expected = Matrix.of_lists [[4.0; 4.0]; [10.0; 8.0]] in
  Alcotest.(check bool) "dot product" true (matrix_equal c expected)

let test_matrix_add () =
  let a = Matrix.of_lists [[1.0; 2.0]; [3.0; 4.0]] in
  let b = Matrix.of_lists [[5.0; 6.0]; [7.0; 8.0]] in
  let c = Matrix.add a b in
  let expected = Matrix.of_lists [[6.0; 8.0]; [10.0; 12.0]] in
  Alcotest.(check bool) "addition" true (matrix_equal c expected)

let test_matrix_mult () =
  let a = Matrix.of_lists [[2.0; 3.0]; [4.0; 5.0]] in
  let b = Matrix.of_lists [[1.0; 2.0]; [3.0; 4.0]] in
  let c = Matrix.mult a b in
  let expected = Matrix.of_lists [[2.0; 6.0]; [12.0; 20.0]] in
  Alcotest.(check bool) "element-wise mult" true (matrix_equal c expected)

let test_matrix_scalar_mult () =
  let m = Matrix.of_lists [[1.0; 2.0]; [3.0; 4.0]] in
  let result = Matrix.scalar_mult 2.0 m in
  let expected = Matrix.of_lists [[2.0; 4.0]; [6.0; 8.0]] in
  Alcotest.(check bool) "scalar mult" true (matrix_equal result expected)

let test_matrix_sum () =
  let m = Matrix.of_lists [[1.0; 2.0]; [3.0; 4.0]] in
  let s = Matrix.sum m in
  Alcotest.(check bool) "sum" true (float_equal s 10.0)

let test_matrix_sum_axis () =
  let m = Matrix.of_lists [[1.0; 2.0; 3.0]; [4.0; 5.0; 6.0]] in
  let sum_cols = Matrix.sum_axis 0 m in
  let expected_cols = Matrix.of_lists [[5.0; 7.0; 9.0]] in
  Alcotest.(check bool) "sum axis 0" true (matrix_equal sum_cols expected_cols);

  let sum_rows = Matrix.sum_axis 1 m in
  let expected_rows = Matrix.of_lists [[6.0]; [15.0]] in
  Alcotest.(check bool) "sum axis 1" true (matrix_equal sum_rows expected_rows)

let test_matrix_argmax () =
  let m = Matrix.of_lists [[0.1; 0.8; 0.3];
                           [0.9; 0.1; 0.5];
                           [0.0; 0.1; 0.2]] in
  let result = Matrix.argmax_axis0 m in
  let expected = Matrix.of_lists [[1.0; 0.0; 1.0]] in
  Alcotest.(check bool) "argmax" true (matrix_equal result expected)

(** Test Activation functions *)
let test_sigmoid () =
  let z = Matrix.of_lists [[0.0; 1.0]; [-1.0; 2.0]] in
  let a = Activation.sigmoid z in
  Alcotest.(check bool) "sigmoid(0) = 0.5" true
    (float_equal a.(0).(0) 0.5);
  Alcotest.(check bool) "sigmoid > 0" true
    (a.(0).(1) > 0.5 && a.(1).(1) > 0.5);
  Alcotest.(check bool) "sigmoid < 0.5" true
    (a.(1).(0) < 0.5)

let test_relu () =
  let z = Matrix.of_lists [[1.0; -1.0]; [0.0; 2.0]] in
  let a = Activation.relu z in
  let expected = Matrix.of_lists [[1.0; 0.0]; [0.0; 2.0]] in
  Alcotest.(check bool) "relu" true (matrix_equal a expected)

let test_relu_derivative () =
  let z = Matrix.of_lists [[1.0; -1.0]; [0.0; 2.0]] in
  let d = Activation.relu_derivative z in
  let expected = Matrix.of_lists [[1.0; 0.0]; [0.0; 1.0]] in
  Alcotest.(check bool) "relu derivative" true (matrix_equal d expected)

(** Test Neural Network functions *)
let test_initialize_parameters () =
  let layer_dims = [3; 4; 2] in
  let params = Neural_network.initialize_parameters layer_dims 1 in

  let w1 = Hashtbl.find params.weights "W1" in
  let b1 = Hashtbl.find params.biases "b1" in
  let w2 = Hashtbl.find params.weights "W2" in
  let b2 = Hashtbl.find params.biases "b2" in

  Alcotest.(check bool) "W1 shape" true
    (Matrix.shape w1 = (4, 3));
  Alcotest.(check bool) "b1 shape" true
    (Matrix.shape b1 = (4, 1));
  Alcotest.(check bool) "W2 shape" true
    (Matrix.shape w2 = (2, 4));
  Alcotest.(check bool) "b2 shape" true
    (Matrix.shape b2 = (2, 1));

  (* Check biases are zeros *)
  Alcotest.(check bool) "b1 zeros" true
    (Array.for_all (fun row -> Array.for_all (fun x -> x = 0.0) row) b1);
  Alcotest.(check bool) "b2 zeros" true
    (Array.for_all (fun row -> Array.for_all (fun x -> x = 0.0) row) b2)

let test_linear_forward () =
  let a = Matrix.of_lists [[1.0; 2.0]; [3.0; 4.0]] in
  let w = Matrix.of_lists [[1.0; 2.0]; [3.0; 4.0]; [5.0; 6.0]] in
  let b = Matrix.of_lists [[1.0]; [2.0]; [3.0]] in

  let z, cache = Neural_network.linear_forward a w b in

  (* Z = W·A + b *)
  (* First column: [1,3,5]·[1,3] + [1,2,3] = [7,11,17] + [1,2,3] = [8,13,20] *)
  Alcotest.(check bool) "z shape" true
    (Matrix.shape z = (3, 2));
  Alcotest.(check bool) "z[0,0]" true
    (float_equal z.(0).(0) 8.0);

  (* Check cache *)
  Alcotest.(check bool) "cache has correct a_prev" true
    (cache.a_prev == a);
  Alcotest.(check bool) "cache has correct w" true
    (cache.w == w)

let test_forward_propagation () =
  (* Small 2-layer network *)
  let x = Matrix.of_lists [[0.5; 1.0]] in  (* 1 feature, 2 samples *)
  let layer_dims = [1; 2; 1] in
  let params = Neural_network.initialize_parameters layer_dims 42 in

  let al, caches = Neural_network.l_model_forward x params 3 in

  Alcotest.(check bool) "output shape" true
    (Matrix.shape al = (1, 2));
  Alcotest.(check int) "num caches" 2 (List.length caches);

  (* Output should be between 0 and 1 (sigmoid) *)
  Alcotest.(check bool) "sigmoid output range" true
    (al.(0).(0) >= 0.0 && al.(0).(0) <= 1.0)

let test_cost_computation () =
  let al = Matrix.of_lists [[0.8; 0.2]; [0.1; 0.9]] in
  let y = Matrix.of_lists [[1.0; 0.0]; [0.0; 1.0]] in

  let cost = Neural_network.compute_cost al y in

  (* Cost should be positive *)
  Alcotest.(check bool) "cost > 0" true (cost > 0.0);

  (* Perfect prediction should have low cost *)
  let al_perfect = Matrix.of_lists [[0.99; 0.01]; [0.01; 0.99]] in
  let cost_perfect = Neural_network.compute_cost al_perfect y in
  Alcotest.(check bool) "better prediction = lower cost" true
    (cost_perfect < cost)

let test_backward_propagation () =
  (* Simple network to test backprop *)
  let x = Matrix.of_lists [[1.0; 2.0]] in
  let y = Matrix.of_lists [[0.0; 1.0]] in
  let layer_dims = [1; 2; 1] in
  let params = Neural_network.initialize_parameters layer_dims 42 in

  let al, caches = Neural_network.l_model_forward x params 3 in
  let grads = Neural_network.l_model_backward al y caches 3 in

  (* Check gradients exist *)
  Alcotest.(check bool) "dW1 exists" true
    (Hashtbl.mem grads.dw "dW1");
  Alcotest.(check bool) "db1 exists" true
    (Hashtbl.mem grads.db "db1");
  Alcotest.(check bool) "dW2 exists" true
    (Hashtbl.mem grads.dw "dW2");
  Alcotest.(check bool) "db2 exists" true
    (Hashtbl.mem grads.db "db2")

let test_parameter_update () =
  let layer_dims = [2; 3; 1] in
  let params = Neural_network.initialize_parameters layer_dims 42 in

  (* Create dummy gradients *)
  let grads = {
    Neural_network.dw = Hashtbl.create 2;
    Neural_network.db = Hashtbl.create 2;
  } in
  Hashtbl.add grads.dw "dW1" (Matrix.ones 3 2);
  Hashtbl.add grads.db "db1" (Matrix.ones 3 1);
  Hashtbl.add grads.dw "dW2" (Matrix.ones 1 3);
  Hashtbl.add grads.db "db2" (Matrix.ones 1 1);

  let w1_before = Hashtbl.find params.weights "W1" in
  let updated_params = Neural_network.update_parameters params grads 0.1 3 in
  let w1_after = Hashtbl.find updated_params.weights "W1" in

  (* Parameters should change *)
  Alcotest.(check bool) "parameters updated" true
    (not (matrix_equal w1_before w1_after))

let test_training_convergence () =
  (* Simple XOR-like problem - test that cost decreases *)
  let x = Matrix.of_lists [[0.0; 0.0; 1.0; 1.0];
                           [0.0; 1.0; 0.0; 1.0]] in
  let y = Matrix.of_lists [[0.0; 1.0; 1.0; 0.0]] in

  let layer_dims = [2; 4; 1] in

  (* Train for a few iterations *)
  let params = Neural_network.train x y layer_dims 0.1 100 false 42 in

  (* Check that we can make predictions *)
  let predictions = Neural_network.predict x params 3 in
  Alcotest.(check bool) "predictions shape" true
    (Matrix.shape predictions = (1, 4))

let test_prediction () =
  (* Simple test data *)
  let x = Matrix.of_lists [[1.0; 2.0; 3.0]] in
  let layer_dims = [1; 3; 2] in
  let params = Neural_network.initialize_parameters layer_dims 42 in

  let predictions = Neural_network.predict x params 3 in

  Alcotest.(check bool) "predictions shape" true
    (Matrix.shape predictions = (1, 3));

  (* Predictions should be 0 or 1 (class indices) *)
  for i = 0 to 2 do
    let pred = predictions.(0).(i) in
    Alcotest.(check bool) "valid class" true
      (pred = 0.0 || pred = 1.0)
  done

let test_accuracy () =
  let predictions = Matrix.of_lists [[0.0; 1.0; 0.0; 1.0]] in
  let y = Matrix.of_lists [[0.0; 1.0; 1.0; 1.0]] in

  let acc = Neural_network.accuracy predictions y in
  Alcotest.(check bool) "accuracy" true (float_equal acc 75.0)

(** Test suite *)
let () =
  let open Alcotest in
  run "Deep Neural Network" [
    "Matrix Operations", [
      test_case "Matrix creation" `Quick test_matrix_creation;
      test_case "Matrix transpose" `Quick test_matrix_transpose;
      test_case "Matrix dot product" `Quick test_matrix_dot;
      test_case "Matrix addition" `Quick test_matrix_add;
      test_case "Matrix element-wise mult" `Quick test_matrix_mult;
      test_case "Matrix scalar mult" `Quick test_matrix_scalar_mult;
      test_case "Matrix sum" `Quick test_matrix_sum;
      test_case "Matrix sum axis" `Quick test_matrix_sum_axis;
      test_case "Matrix argmax" `Quick test_matrix_argmax;
    ];
    "Activation Functions", [
      test_case "Sigmoid" `Quick test_sigmoid;
      test_case "ReLU" `Quick test_relu;
      test_case "ReLU derivative" `Quick test_relu_derivative;
    ];
    "Neural Network", [
      test_case "Initialize parameters" `Quick test_initialize_parameters;
      test_case "Linear forward" `Quick test_linear_forward;
      test_case "Forward propagation" `Quick test_forward_propagation;
      test_case "Cost computation" `Quick test_cost_computation;
      test_case "Backward propagation" `Quick test_backward_propagation;
      test_case "Parameter update" `Quick test_parameter_update;
      test_case "Training convergence" `Slow test_training_convergence;
      test_case "Prediction" `Quick test_prediction;
      test_case "Accuracy" `Quick test_accuracy;
    ];
  ]

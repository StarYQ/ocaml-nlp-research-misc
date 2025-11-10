(** Demo application for digit recognition neural network *)

open Digit_recognition_nn

(** Create synthetic digit data for demonstration
    This creates simplified patterns for digits 0-9 *)
let create_synthetic_digit_data num_samples =
  Random.init 42;

  (* Create 64 features (8x8 pixels) for each sample *)
  (* We'll create simple patterns for each digit *)
  let x_data = Array.make_matrix 64 num_samples 0.0 in
  let y_data = Array.make_matrix 1 num_samples 0.0 in

  for i = 0 to num_samples - 1 do
    let digit = i mod 10 in
    y_data.(0).(i) <- float_of_int digit;

    (* Create simple patterns with noise *)
    for j = 0 to 63 do
      let base_value = match digit with
        | 0 -> if j / 8 = 0 || j / 8 = 7 || j mod 8 = 0 || j mod 8 = 7 then 0.8 else 0.1
        | 1 -> if j mod 8 = 4 then 0.8 else 0.1
        | 2 -> if j / 8 = 0 || j / 8 = 3 || j / 8 = 7 then 0.8 else 0.1
        | 3 -> if j / 8 = 0 || j / 8 = 3 || j / 8 = 7 || j mod 8 = 7 then 0.8 else 0.1
        | 4 -> if j mod 8 = 0 || j / 8 = 3 || j mod 8 = 7 then 0.8 else 0.1
        | 5 -> if j / 8 = 0 || j / 8 = 3 || j / 8 = 7 || j mod 8 = 0 then 0.8 else 0.1
        | 6 -> if j / 8 = 0 || j / 8 = 3 || j / 8 = 7 || j mod 8 = 0 then 0.8 else 0.1
        | 7 -> if j / 8 = 0 || j mod 8 = 7 then 0.8 else 0.1
        | 8 -> if j / 8 = 0 || j / 8 = 3 || j / 8 = 7 || j mod 8 = 0 || j mod 8 = 7 then 0.8 else 0.1
        | 9 -> if j / 8 = 0 || j / 8 = 3 || j mod 8 = 7 then 0.8 else 0.1
        | _ -> 0.1
      in
      (* Add small noise *)
      let noise = (Random.float 0.2) -. 0.1 in
      x_data.(j).(i) <- base_value +. noise
    done
  done;

  (x_data, y_data)

(** Convert labels to one-hot encoding *)
let one_hot_encode y num_classes =
  let _, m = Matrix.shape y in
  let y_encoded = Matrix.zeros num_classes m in

  for i = 0 to m - 1 do
    let label = int_of_float y.(0).(i) in
    y_encoded.(label).(i) <- 1.0
  done;

  y_encoded

(** Print a visual representation of an 8x8 digit *)
let print_digit x_col =
  Printf.printf "\nDigit visualization (8x8):\n";
  for i = 0 to 7 do
    for j = 0 to 7 do
      let idx = i * 8 + j in
      let val_ = x_col.(idx).(0) in
      if val_ > 0.5 then Printf.printf "██"
      else if val_ > 0.3 then Printf.printf "▓▓"
      else if val_ > 0.1 then Printf.printf "░░"
      else Printf.printf "  "
    done;
    Printf.printf "\n"
  done;
  Printf.printf "\n"

(** Main demo *)
let () =
  Printf.printf "=================================================\n";
  Printf.printf "Deep Neural Network - Digit Recognition Demo\n";
  Printf.printf "=================================================\n\n";

  (* Configuration *)
  let num_train_samples = 500 in
  let num_test_samples = 100 in
  let learning_rate = 0.01 in
  let num_iterations = 5000 in
  let num_classes = 10 in

  (* Architecture: 64 → 60 → 10 → 10 (matching the Medium article) *)
  let layer_dims = [64; 60; 10; num_classes] in

  Printf.printf "Network Architecture:\n";
  Printf.printf "  Input layer:    64 neurons (8x8 pixels)\n";
  Printf.printf "  Hidden layer 1: 60 neurons (ReLU)\n";
  Printf.printf "  Hidden layer 2: 10 neurons (ReLU)\n";
  Printf.printf "  Output layer:   10 neurons (Sigmoid)\n\n";

  Printf.printf "Hyperparameters:\n";
  Printf.printf "  Learning rate:  %.4f\n" learning_rate;
  Printf.printf "  Iterations:     %d\n" num_iterations;
  Printf.printf "  Training size:  %d samples\n" num_train_samples;
  Printf.printf "  Test size:      %d samples\n\n" num_test_samples;

  (* Generate training data *)
  Printf.printf "Generating synthetic digit data...\n";
  let x_train, y_train = create_synthetic_digit_data num_train_samples in
  let y_train_encoded = one_hot_encode y_train num_classes in

  (* Generate test data *)
  let x_test, y_test = create_synthetic_digit_data num_test_samples in
  let _y_test_encoded = one_hot_encode y_test num_classes in

  Printf.printf "Training data shape: (%d, %d)\n" (fst (Matrix.shape x_train)) (snd (Matrix.shape x_train));
  Printf.printf "Training labels shape: (%d, %d)\n\n" (fst (Matrix.shape y_train_encoded)) (snd (Matrix.shape y_train_encoded));

  (* Train the model *)
  Printf.printf "Training neural network...\n";
  Printf.printf "--------------------------------------------\n";
  let params = Neural_network.train
    x_train
    y_train_encoded
    layer_dims
    learning_rate
    num_iterations
    true  (* print_cost *)
    1     (* seed *)
  in
  Printf.printf "--------------------------------------------\n";
  Printf.printf "Training complete!\n\n";

  (* Evaluate on training set *)
  Printf.printf "Evaluating on training set...\n";
  let train_predictions = Neural_network.predict x_train params (List.length layer_dims) in
  let train_accuracy = Neural_network.accuracy train_predictions y_train in
  Printf.printf "Training Accuracy: %.2f%%\n\n" train_accuracy;

  (* Evaluate on test set *)
  Printf.printf "Evaluating on test set...\n";
  let test_predictions = Neural_network.predict x_test params (List.length layer_dims) in
  let test_accuracy = Neural_network.accuracy test_predictions y_test in
  Printf.printf "Test Accuracy: %.2f%%\n\n" test_accuracy;

  (* Show some predictions *)
  Printf.printf "Sample Predictions:\n";
  Printf.printf "--------------------------------------------\n";
  for i = 0 to min 9 (num_test_samples - 1) do
    let x_sample = Array.init 64 (fun j -> [| x_test.(j).(i) |]) in
    print_digit x_sample;
    Printf.printf "True label:      %d\n" (int_of_float y_test.(0).(i));
    Printf.printf "Predicted label: %d\n" (int_of_float test_predictions.(0).(i));
    Printf.printf "--------------------------------------------\n"
  done;

  Printf.printf "\nDemo complete!\n"

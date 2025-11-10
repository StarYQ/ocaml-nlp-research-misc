(** Training digit recognition neural network on REAL handwritten digit images *)

open Digit_recognition_nn

(** Convert labels to one-hot encoding *)
let one_hot_encode y num_classes =
  let _, m = Matrix.shape y in
  let y_encoded = Matrix.zeros num_classes m in

  for i = 0 to m - 1 do
    let label = int_of_float y.(0).(i) in
    y_encoded.(label).(i) <- 1.0
  done;

  y_encoded

(** Print a visual representation of a 64x64 digit *)
let print_digit x_col =
  Printf.printf "\nDigit visualization (64x64):\n";
  for i = 0 to 63 do
    for j = 0 to 63 do
      let idx = i * 64 + j in
      let val_ = x_col.(idx).(0) in
      (* Different threshold since we're using standardized values *)
      if val_ > 0.5 then Printf.printf "██"
      else if val_ > 0.0 then Printf.printf "▓▓"
      else if val_ > -0.5 then Printf.printf "░░"
      else Printf.printf "  "
    done;
    Printf.printf "\n"
  done;
  Printf.printf "\n"

(** Main training on real digit images *)
let () =
  Printf.printf "===========================================================\n";
  Printf.printf "Deep Neural Network - REAL Digit Recognition Training\n";
  Printf.printf "===========================================================\n\n";

  (* Configuration *)
  let archive_path = "images/digit_imgs/archive" in
  let learning_rate = 0.01 in
  let num_iterations = 5000 in
  let num_classes = 10 in

  (* Architecture: 4096 → 256 → 128 → 10 *)
  let layer_dims = [4096; 256; 128; num_classes] in

  Printf.printf "Network Architecture:\n";
  Printf.printf "  Input layer:    4096 neurons (64x64 pixels)\n";
  Printf.printf "  Hidden layer 1: 256 neurons (ReLU)\n";
  Printf.printf "  Hidden layer 2: 128 neurons (ReLU)\n";
  Printf.printf "  Output layer:   10 neurons (Sigmoid)\n\n";

  Printf.printf "Hyperparameters:\n";
  Printf.printf "  Learning rate:  %.4f\n" learning_rate;
  Printf.printf "  Iterations:     %d\n" num_iterations;
  Printf.printf "  Train/test:     80/20 split\n\n";

  (* Load and prepare dataset *)
  let x_train, y_train, x_test, y_test =
    Image_loader.load_dataset
      ~train_ratio:0.8
      ~seed:42
      ~standardize_data:true
      archive_path
  in

  (* One-hot encode labels *)
  let y_train_encoded = one_hot_encode y_train num_classes in
  let y_test_encoded = one_hot_encode y_test num_classes in

  Printf.printf "Data shapes:\n";
  Printf.printf "  X_train: (%d, %d)\n" (fst (Matrix.shape x_train)) (snd (Matrix.shape x_train));
  Printf.printf "  Y_train: (%d, %d)\n" (fst (Matrix.shape y_train_encoded)) (snd (Matrix.shape y_train_encoded));
  Printf.printf "  X_test:  (%d, %d)\n" (fst (Matrix.shape x_test)) (snd (Matrix.shape x_test));
  Printf.printf "  Y_test:  (%d, %d)\n\n" (fst (Matrix.shape y_test_encoded)) (snd (Matrix.shape y_test_encoded));

  (* Train the model *)
  Printf.printf "Training neural network on real images...\n";
  Printf.printf "------------------------------------------------------------\n";
  let params = Neural_network.train
    x_train
    y_train_encoded
    layer_dims
    learning_rate
    num_iterations
    true  (* print_cost *)
    1     (* seed *)
  in
  Printf.printf "------------------------------------------------------------\n";
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
  Printf.printf "Sample Predictions from Test Set:\n";
  Printf.printf "------------------------------------------------------------\n";
  let num_test_samples = snd (Matrix.shape x_test) in
  for i = 0 to min 9 (num_test_samples - 1) do
    let x_sample = Array.init 4096 (fun j -> [| x_test.(j).(i) |]) in
    print_digit x_sample;
    Printf.printf "True label:      %d\n" (int_of_float y_test.(0).(i));
    Printf.printf "Predicted label: %d\n" (int_of_float test_predictions.(0).(i));
    Printf.printf "------------------------------------------------------------\n"
  done;

  Printf.printf "\nTraining on real images complete!\n";
  Printf.printf "\nSummary:\n";
  Printf.printf "  Training samples: %d\n" (snd (Matrix.shape x_train));
  Printf.printf "  Test samples:     %d\n" (snd (Matrix.shape x_test));
  Printf.printf "  Training accuracy: %.2f%%\n" train_accuracy;
  Printf.printf "  Test accuracy:     %.2f%%\n" test_accuracy

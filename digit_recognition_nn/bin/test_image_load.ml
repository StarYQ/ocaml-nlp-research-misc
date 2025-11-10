(** Quick test to verify image loading works *)

open Digit_recognition_nn

let () =
  Printf.printf "Testing image loading...\n\n";

  (* Test loading a single image *)
  let test_file = "images/digit_imgs/archive/0/Zero_full (1).jpg" in
  Printf.printf "Loading single image: %s\n" test_file;

  try
    let pixels = Image_loader.load_and_preprocess_image test_file in
    Printf.printf "Success! Loaded image with %d pixels (64x64 resolution)\n" (Array.length pixels);
    Printf.printf "First 10 pixel values: ";
    for i = 0 to min 9 (Array.length pixels - 1) do
      Printf.printf "%.3f " pixels.(i)
    done;
    Printf.printf "\n\n";

    (* Test loading a few images from directory *)
    Printf.printf "Loading images from directory 0...\n";
    let images = Image_loader.load_images_from_dir "images/digit_imgs/archive/0" in
    Printf.printf "Loaded %d images\n\n" (List.length images);

    Printf.printf "Image loading test PASSED!\n"
  with
  | e ->
      Printf.printf "ERROR: %s\n" (Printexc.to_string e);
      Printf.printf "Image loading test FAILED!\n";
      exit 1

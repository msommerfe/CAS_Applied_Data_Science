import OCR_from_scratch
import util


model = OCR_from_scratch.build_model()


#############HIER NOCH WEITER MACHEN UND VALIDATION DATASET ORDENTLICH AUFBEREITEN. AM BESTEN FUNKTION SCHREIBEN, DIE DIE dATEN AUFBEREITET
keyVal100k = util.import_txt_csv_label_file(path = "/mnt/c/dev/datasets/OCR/tr_synth_100K_cropped/annotations.txt")
keyVal100k = util.make_total_path_for_all_image_names(keyVal100k, path= '/mnt/c/dev/datasets/OCR/tr_synth_100K_cropped/images/')
key_val = util.delete_key_values_that_not_in_alphabet(key_val)
final_key_val= util.delete_key_values_that_have_a_too_long_label(key_val)
final_key_val = util.delete_key_values_with_too_small_aspect_ratio(final_key_val)


model.load_weights('/mnt/c/dev/git/CAS_Applied_Data_Science/CAS_Final_Project/Weights/100k_batch256_alphaAll.weights.h5')
score = model.evaluate(validation_dataset, verbose=0)
print('Score: ' +str(score))
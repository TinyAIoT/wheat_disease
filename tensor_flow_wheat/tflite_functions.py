import tensorflow as tf
import numpy as np

# Convert model to tflite
def convert_tflite_model(model):
    print("Converting ....")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    
    tflite_model = converter.convert()
    print("Converted ....")
    return tflite_model

def save_tflite_model(tflite_model, save_dir, model_name):
	print("Saving ...")
	import os
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	save_path = os.path.join(save_dir, model_name)
	print("Tflite save path:",save_path)
	with open(save_path, "wb") as f:
		f.write(tflite_model)
	print("Tflite model saved to :", save_dir)
 
def test_tflite(tflite_model,X_test,y_test):
	interpreter = tf.lite.Interpreter(model_content=tflite_model)
	interpreter.allocate_tensors()
	# Get input and output tensors
	input_tensor_index = interpreter.get_input_details()[0]['index']
	output_tensor_index = interpreter.get_output_details()[0]['index']
	# Run inference on test data
	true_pos = 0
	true_neg = 0
	false_pos = 0
	false_neg = 0
	total_predictions = len(X_test)

	for i in range(total_predictions):
		input_data = np.expand_dims(X_test[i], axis=0).astype(np.float32)
		interpreter.set_tensor(input_tensor_index, input_data)
		interpreter.invoke()
		lite_predictions = interpreter.get_tensor(output_tensor_index)[0]
		
		# Compare predicted label with true label
		predicted_label = round(lite_predictions[0],0)
		true_label = y_test[i]
		
		if predicted_label == true_label:
			if predicted_label == 1:
				true_pos += 1
			else:
				true_neg += 1
		else:
			if predicted_label == 1:
				false_pos += 1
			else:
				false_neg += 1
    		#else:
			#	print("lite model predicted: ", lite_predictions[0])
			#	print("org model predicted:  ", tflite_model.predict(np.array( [X_test[i],]),verbose = 0)[0][0])
			

	# Compute accuracy
	print("TensorFlow Lite model:")
	accuracy = (true_pos+true_neg) / total_predictions
	if (true_pos + false_pos) > 0:
		precision = (true_pos) / (true_pos + false_pos)
	else:
		precision = 0.0
	if (true_pos + false_neg) > 0:
		recall = (true_pos) / (true_pos + false_neg)
	else:
		recall = 0.0
	return accuracy,precision,recall
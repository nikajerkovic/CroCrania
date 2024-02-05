from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
import os

app = Flask(__name__)

# Define the required columns
required_columns = ['GOL', 'NOL', 'XCB', 'ZYB', 'BBH', 'AUB', 'WFB', 'NLH_L', 'NLH_R', 'OBB_L', 'OBH_R', 'FRC', 'PAC', 'OCC', 'FOL', 'FOB', 'MDH_L', 'ZOB']

alternative_columns = ['ZYB', 'NLH_L', 'OBH_R']

non_standard_features = ['ast_l_b', 'ast_l_ms_l', 'b_po_r', 'ba_zma_l', 'en_ba_ec_l', 'foram_l_ec_r',
                     'ft_l_ft_r', 'ftm_l_ft_r', 'g_ns_r', 'n_ns_l', 'ns_r_al_r', 'o_ec_r', 'orb_d_l_ec_r',
                     'orb_u_l_orb_d_r', 'ra_l_ra_r', 'ra_r_b', 'ra_r_ms_l', 'zo_l_ec_l', 'zo_l_ftm_r',
                     'zy_l_ms_r', 'zy_l_zy_r', 'zy_r_b', 'zy_r_d_l']

def preprocessDataAndPredict(feature_dict):
    # Extract the features from the provided dictionary

    print(feature_dict)


    if len(feature_dict) == 3 and all(key in feature_dict for key in alternative_columns):
        features = [float(feature_dict[column]) for column in alternative_columns]
        file = open("best_lr_model.pkl", "rb")
        scaled = True
        scaler =joblib.load('Standard_scaler_3.bin')
    elif len(feature_dict) == 18 and all(key in feature_dict for key in required_columns):
        features = [float(feature_dict[column]) for column in required_columns]
        file = open("best_svm_model_new.pkl", "rb")
        scaled = True
        scaler =joblib.load('Standard_scaler_18.bin')
    elif len(feature_dict) == 23 and all(key in feature_dict for key in non_standard_features):
        features = [float(feature_dict[column]) for column in non_standard_features]
        file = open("best_lda_model.pkl", "rb")
        scaled = False
    elif len(feature_dict) == 99:
        features = [float(value) for value in feature_dict.values()]
        file = open("lda_99.pkl", "rb")
        scaled = True
        scaler =joblib.load('Standard_scaler_99.bin')
    
    #scaler = StandardScaler()

    test_array = np.array(features).reshape(1, -1)
    print(test_array)

    if scaled:
        test_array = scaler.transform(test_array)
        print(test_array)

    # Load trained model
    trained_model = joblib.load(file)

    y_pred_prob_test = trained_model.predict_proba(test_array)

    # Get the probability estimates for both classes
    female_probabilities = y_pred_prob_test[:, 0]  # Assuming female is class 0
    male_probabilities = y_pred_prob_test[:, 1]    # Assuming male is class 1

    predict = trained_model.predict(test_array)

    prediction_label = 'Male' if predict[0] == 'M' else 'Female'


    # Create a dictionary to hold the results
    result_dict = {
        'Prediction': prediction_label,
        'Female Probability': female_probabilities[0],
        'Male Probability': male_probabilities[0]
    }

    print(result_dict)

    # Add 'id' and 'filename' to the result dictionary if they are present in the input
    if 'id' in feature_dict:
        result_dict['ID'] = feature_dict['id']

    if 'filename' in feature_dict:
        result_dict['Filename'] = feature_dict['filename']

    return result_dict

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        # Check if the request contains a file
        try:
            print("In try")
            if 'excelFile' in request.files:
                print("in excel")
                # Process the file (you can add your logic here)
                excel_file = request.files['excelFile']
                # Assuming pandas is used for processing the Excel file
                df = pd.read_excel(excel_file)
                pd.set_option('display.max_columns', None)

                print(df)

                if all(col == expected_col for col, expected_col in zip(df.columns[-3:], alternative_columns)):
                        selected_columns = alternative_columns
                elif all(col == expected_col for col, expected_col in zip(df.columns[-18:], required_columns)):
                        selected_columns = required_columns
                else:
                    print("MISSING")
                    return jsonify({'error': 'Missing required columns. Please refer to template.'})

                numeric_check = True

                for col in selected_columns:
                    try:
                        pd.to_numeric(df[col], errors='raise')
                    except ValueError:
                        numeric_check = False
                        break

                if not numeric_check:
                    return jsonify({'error': 'Please use only numeric values for measurments.'})

                empty_columns = []

                for col in selected_columns:
                    if df[col].isna().all():
                        empty_columns.append(col)
                if empty_columns:
                    return jsonify({'error': 'Please fill all the columns.'})

                                
                # Extract the required columns for prediction
                features_df = df[selected_columns]
                prediction_results = []

                for index, row in features_df.iterrows():
                    result = preprocessDataAndPredict(row.to_dict())
                    prediction_results.append(result)

                for index, row in df.iterrows():
                    row_dict = row.to_dict()
                    if 'FILENAME' in row_dict:
                        value = row_dict['FILENAME']
                        prediction_results[index]['Filename'] = value
                    if 'ID' in row_dict:
                        value = row_dict['ID']
                        prediction_results[index]['ID'] = value

                print(prediction_results)
                return jsonify(prediction_results)
        except:
            # Process the form data
            to_predict_dict = request.form.to_dict()
            filtered_dict = {key: value for key, value in to_predict_dict.items() if value}

            try:
                prediction_results = preprocessDataAndPredict(filtered_dict)
                return jsonify([prediction_results])
            except ValueError:
                return jsonify({'error': 'Please enter valid values'})


landmark_names = {
    1: 'g', 2: 'op', 3: 'n', 4: 'op_n', 5: 'eu_l', 6: 'eu_r', 7: 'zy_l', 8: 'zy_r',
    9: 'ra_l', 10: 'ra_r', 11: 'ast_l', 12: 'ast_r', 13: 'ba', 14: 'en_ba',
    15: 'o', 16: 'foram_l', 17: 'foram_r', 18: 'alv', 19: 'ecm_l', 20: 'ecm_r',
    21: 'pr', 22: 'ns_l', 23: 'ns_r', 24: 'al_l', 25: 'al_r', 26: 'zma_l',
    27: 'zma_r', 28: 'zo_l', 29: 'zo_r', 30: 'ftm_l', 31: 'ftm_r', 32: 'ft_l',
    33: 'ft_r', 34: 'b', 35: 'l', 36: 'po_l', 37: 'ms_l', 38: 'po_r', 39: 'ms_r',
    40: 'd_l', 41: 'ec_l', 42: 'orb_d_l', 43: 'orb_u_l', 44: 'd_r', 45: 'ec_r',46: 'orb_d_r', 47: 'orb_u_r'
}
inverse_landmark_names = {v: k for k, v in landmark_names.items()}

# Define the landmark pairs list
landmark_pairs = [
    ("al_l", "al_r"), ("al_l", "zma_r"), ("al_r", "ftm_r"), ("al_r", "ms_l"),
    ("alv", "al_l"), ("alv", "zma_r"), ("ast_l", "ba"), ("ast_l", "ms_l"),
    ("ast_l", "ms_r"), ("ast_r", "ec_l"), ("ast_r", "ft_r"), ("ast_r", "ms_r"),
    ("ast_r", "o"), ("ast_r", "po_r"), ("b", "l"), ("ba", "alv"),
    ("d_l", "orb_d_l"), ("d_l", "orb_u_r"), ("d_r", "ec_r"), ("ec_l", "orb_u_l"),
    ("ec_r", "orb_d_r"), ("eu_l", "ra_l"), ("eu_l", "zy_r"), ("eu_r", "ast_r"),
    ("eu_r", "ft_l"), ("eu_r", "l"), ("foram_l", "foram_r"), ("foram_r", "ec_l"),
    ("foram_r", "ms_l"), ("foram_r", "ms_r"), ("ft_r", "ec_l"), ("ft_r", "ms_l"),
    ("ftm_l", "orb_d_l"), ("ftm_r", "orb_u_r"), ("g", "al_l"), ("g", "ast_l"),
    ("g", "d_l"), ("g", "d_r"), ("g", "ec_r"), ("g", "eu_l"), ("g", "ftm_l"),
    ("g", "n"), ("g", "orb_u_l"), ("g", "zo_l"), ("g", "zo_r"), ("g", "zy_r"),
    ("l", "ms_l"), ("n", "alv"), ("n", "ba"), ("n", "d_l"), ("n", "ns_l"),
    ("n", "zma_l"), ("ns_l", "al_r"), ("ns_l", "d_r"), ("ns_l", "orb_d_r"),
    ("ns_r", "al_l"), ("ns_r", "al_r"), ("o", "foram_r"), ("o", "l"),
    ("o", "po_r"), ("o", "zma_l"), ("op_n", "ast_l"), ("op_n", "ast_r"),
    ("op_n", "ra_r"), ("orb_d_l", "orb_u_l"), ("orb_u_l", "d_r"), ("po_l", "d_l"),
    ("po_r", "d_l"), ("po_r", "ms_r"), ("ra_l", "ast_l"), ("ra_l", "ms_l"),
    ("ra_l", "orb_d_r"), ("ra_l", "po_l"), ("ra_l", "po_r"), ("ra_r", "foram_l"),
    ("ra_r", "ft_l"), ("ra_r", "ft_r"), ("ra_r", "l"), ("ra_r", "orb_u_r"), ("zma_l", "zo_l"),
    ("zo_l", "d_r"), ("zo_l", "ec_l"), ("zo_l", "ec_r"), ("zo_l", "ftm_l"),
    ("zo_l", "ftm_r"), ("zo_l", "orb_d_r"), ("zo_r", "ec_r"), ("zo_r", "ftm_r"),
    ("zy_l", "al_l"), ("zy_l", "foram_l"), ("zy_l", "ms_r"), ("zy_l", "po_l"),
    ("zy_r", "alv"), ("zy_r", "ast_r"), ("zy_r", "ba"), ("zy_r", "ec_r"),
    ("zy_r", "ns_r"), ("zy_r", "po_r"), ("zy_r", "zo_r")
]

def read_data_from_nts(file):
    try:
        data = file.readlines()[2:]  # Skip the first two lines
        landmarks = [list(map(float, line.split())) for line in data]
        landmarks = [coord if coord != 9999 else np.nan for coords in landmarks for coord in coords]
        landmarks = np.array(landmarks).reshape(-1, 3)
        return landmarks
    except Exception as e:
        print(f"Error reading file {file.filename}: {e}")
        return None

def  read_data_from_csv(file):
    try:
        df = pd.read_csv(file, skiprows=[0])  # Skip the first row with general info
        df = df.rename(columns={'#': 'Landmark', 'Name': 'LandmarkName', 'X': 'X', 'Y': 'Y', 'Z': 'Z'})
        landmarks = df[['Landmark', 'X', 'Y', 'Z']].values
        return landmarks
    except Exception as e:
        print(f"Error reading CSV file {file}: {e}")
        return None

def read_data_from_pts(file):
    try:
        lines = file.readlines()
        num_landmarks = int(lines[1].strip())
        landmarks = []
        for line in lines[2:2 + num_landmarks]:
            parts = line.split()
            coords = [float(coord) if coord != '9999' else np.nan for coord in parts[1:]]  # Exclude the landmark name/number
            landmarks.append(coords)
        landmarks = np.array(landmarks)
        return landmarks
    except Exception as e:
        print(f"Error reading PTS file {file}: {e}")
        return None


def read_data_from_morphologika(file):
    try:
        lines = file.readlines()
        print(lines)
        num_landmarks_index = lines.index( b'[landmarks]\r\n') + 1
        num_landmarks = int(lines[num_landmarks_index].strip())
        rawpoints_index = lines.index(b'[rawpoints]\r\n') + 1
        landmarks = []
        for i in range(rawpoints_index + 1, rawpoints_index + 1 + num_landmarks):
            coords = lines[i].strip().split()
            coords = [float(coord) if coord != '9999' else np.nan for coord in coords]
            landmarks.append(coords)
        landmarks = np.array(landmarks)
        return landmarks
    except Exception as e:
        print(f"Error reading Morphologika file {file}: {e}")
        return None


def process_uploaded_folder(uploaded_folder):
    all_data = {}
    file_types = set()  # Set to store unique file extensions
    
    # Collect file extensions present in the folder
    for file in uploaded_folder:
        ext = file.filename.split(".")[-1].lower()  # Extract file extension
        file_types.add(ext)
    
    # If there are multiple file extensions, return an empty dictionary
    if len(file_types) != 1:
        return all_data
    
    # Get the single file extension (all files in the folder should have the same extension)
    file_extension = file_types.pop()
    
    # Process files based on their extension
    for file in uploaded_folder:
        if file.filename.lower().endswith("." + file_extension):
            if file_extension == "nts":
                landmarks = read_data_from_nts(file)
            elif file_extension == "pts":
                landmarks = read_data_from_pts(file)
            elif file_extension == "csv":
                landmarks = read_data_from_csv(file)
            elif file_extension == "txt":
                landmarks = read_data_from_morphologika(file)
            else:
                continue  # Skip unsupported file types
            
            if landmarks is not None:
                all_data[file.filename] = landmarks
                
    return all_data


def calculate_distances(all_data):
    specific_pairs_indices = []
    for lm1, lm2 in landmark_pairs:
        index1 = inverse_landmark_names.get(lm1)
        index2 = inverse_landmark_names.get(lm2)
        if index1 and index2:
            specific_pairs_indices.append((index1, index2))

    distance_data = []
    for filename, landmarks in all_data.items():
        if landmarks is not None:
            distances = {'Filename': filename}
            for (i, j) in specific_pairs_indices:
                lm1, lm2 = landmarks[i-1], landmarks[j-1]  # Subtract 1 for zero-based indexing
                distance = np.linalg.norm(lm1 - lm2) if not np.isnan(lm1).any() and not np.isnan(lm2).any() else np.nan
                distances[f'{landmark_names.get(i)}_{landmark_names.get(j)}'] = distance# Multiply by 10 to convert to mm
            distance_data.append(distances)

    distance_df = pd.DataFrame(distance_data)
    return distance_df

@app.route('/predict_non_standard', methods=['POST'])
def predict_non_standard():
    if request.method == "POST":
        folder = request.files.getlist('folder')
        print(folder)
        #if len(folder) >= 1:
        if folder and folder[0].filename:
            #all_nts_files = all(file.filename.lower().endswith('.nts') for file in folder)
            #if all_nts_files:
                #uploaded_folder = request.files['folderUpload']
            all_data = process_uploaded_folder(folder)
            print(all_data)
            try:
                if all_data:
                        distance_df = calculate_distances(all_data)
                        filenames = distance_df['Filename']
                        pd.set_option('display.max_columns', None)
                        print(distance_df)
                        features_df = distance_df.iloc[:, 1:]
                        print("FEATURES")
                        print (features_df)
                        # Proceed with further processing using all_data dictionary
                        prediction_results = []

                        for filename, row in zip(filenames, features_df.iterrows()):
                            index, data = row
                            result = preprocessDataAndPredict(data.to_dict())
                            prediction_result = {'Filename': filename, 'Prediction': result['Prediction']}
                            prediction_result.update(result)  # Merge the result dictionary into prediction_result
                            prediction_results.append(prediction_result)

                        print(prediction_results)
                        return jsonify(prediction_results)

                else:
            # If any file doesn't have the ".nts" extension, return an error message
                    print("nije nts")
                    return jsonify({'error': 'Uploaded folder contains files without allowed extensions'})
            except:
                print("Problem!")
                return jsonify({'error': 'Certain files contents can be processed. Please refer to the provided file examples.'})


                
        else:
            to_predict_dict = request.form.to_dict()
            filtered_dict = {key: value for key, value in to_predict_dict.items() if value}

            try:
                    prediction_results = preprocessDataAndPredict(filtered_dict)
                    return jsonify([prediction_results])
            except ValueError:
                    return jsonify({'error': 'Please enter valid values'})



if __name__ == '__main__':
    app.run(debug=True)
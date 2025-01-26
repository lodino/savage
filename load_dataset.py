import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.impute import KNNImputer
# from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

def process_german(df, preprocess):
    df['status'] = df['status'].map({'A11': 0, 'A12': 1, 'A13': 2, 'A14': 3}).astype(int)
    df['credit_hist'] = df['credit_hist'].map({'A34': 0, 'A33': 1, 'A32': 2, 'A31': 3, 'A30': 4}).astype(int)

    df['savings'] = df['savings'].map({'A61': 0, 'A62': 1, 'A63': 2, 'A64': 3, 'A65': 4}).astype(int)
    df['employment'] = df['employment'].map({'A71': 0, 'A72': 1, 'A73': 2, 'A74': 3, 'A75': 4}).astype(int)
    df['gender'] = df['personal_status'].map({'A91': 1, 'A92': 0, 'A93': 1, 'A94': 1, 'A95': 0}).astype(int)
    df['debtors'] = df['debtors'].map({'A101': 0, 'A102': 1, 'A103': 2}).astype(int)
    df['property'] = df['property'].map({'A121': 3, 'A122': 2, 'A123': 1, 'A124': 0}).astype(int)
    df['install_plans'] = df['install_plans'].map({'A141': 1, 'A142': 1, 'A143': 0}).astype(int)
    if preprocess:
        df = pd.concat([df, pd.get_dummies(df['purpose'], prefix='purpose')],axis=1)
        df = pd.concat([df, pd.get_dummies(df['housing'], prefix='housing')],axis=1)
        df.loc[(df['credit_amt'] <= 2000), 'credit_amt'] = 0
        df.loc[(df['credit_amt'] > 2000) & (df['credit_amt'] <= 5000), 'credit_amt'] = 1
        df.loc[(df['credit_amt'] > 5000), 'credit_amt'] = 2
        df.loc[(df['duration'] <= 12), 'duration'] = 0
        df.loc[(df['duration'] > 12) & (df['duration'] <= 24), 'duration'] = 1
        df.loc[(df['duration'] > 24) & (df['duration'] <= 36), 'duration'] = 2
        df.loc[(df['duration'] > 36), 'duration'] = 3
        df['age'] = df['age'].apply(lambda x : 1 if x >= 45 else 0) # 1 if old, 0 if young
    df['job'] = df['job'].map({'A171': 0, 'A172': 1, 'A173': 2, 'A174': 3}).astype(int)
    df['telephone'] = df['telephone'].map({'A191': 0, 'A192': 1}).astype(int)
    df['foreign_worker'] = df['foreign_worker'].map({'A201': 1, 'A202': 0}).astype(int)

    return df


def process_law(df):
    # df.loc[(df['lsat'] <= 30), 'lsat'] = 0.0
    # df.loc[(df['lsat'] > 30), 'lsat'] = 1.0
    # df.loc[(df['decile3'] <= 5), 'decile3'] = 0.0
    # df.loc[(df['decile3'] > 5), 'decile3'] = 1.0
    # df['decile1b'] = pd.cut(df['decile1b'], bins=3, labels=range(3), include_lowest=True).astype(float)
    # df.loc[(df['ugpa'] <= 3), 'ugpa'] = 0.0
    # df.loc[(df['ugpa'] > 3), 'ugpa'] = 1.0
    # df['zfygpa'] = pd.cut(df['zfygpa'], bins=2, labels=range(2), include_lowest=True).astype(float)
    # df.loc[(df['zgpa'] <= -1), 'zgpa'] = 0.0
    # df.loc[(df['zgpa'] > -1) & (df['zgpa'] <= 1), 'zgpa'] = 1.0
    # df.loc[(df['zgpa'] > 1), 'zgpa'] = 2.0
    return df

def process_adult(df):
    # replace missing values (?) to nan and then drop the columns
    df['country'] = df['country'].replace('?',np.nan)
    df['workclass'] = df['workclass'].replace('?',np.nan)
    df['occupation'] = df['occupation'].replace('?',np.nan)
    # dropping the NaN rows now
    df.dropna(how='any',inplace=True)
    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1}).astype(int)
    df['age'] = df['age'].apply(lambda x : 1 if x >= 45 else 0) # 1 if old, 0 if young
    df['workclass'] = df['workclass'].map({'Never-worked': 0, 'Without-pay': 1, 'State-gov': 2, 'Local-gov': 3, 'Federal-gov': 4, 'Self-emp-inc': 5, 'Self-emp-not-inc': 6, 'Private': 7}).astype(int)
    df['education'] = df['education'].map({'Preschool': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3, '9th': 4, '10th': 5, '11th': 6, '12th': 7, 'HS-grad':8, 'Some-college': 9, 'Bachelors': 10, 'Prof-school': 11, 'Assoc-acdm': 12, 'Assoc-voc': 13, 'Masters': 14, 'Doctorate': 15}).astype(int)
#     df.loc[(df['education'] <= 4), 'education'] = 0
#     df.loc[(df['education'] > 4) & (df['education'] <= 8), 'education'] = 1
#     df.loc[(df['education'] > 8) & (df['education'] <= 13), 'education'] = 2
#     df.loc[(df['education'] > 13), 'education'] = 3
    df['marital'] = df['marital'].map({'Married-civ-spouse': 2, 'Divorced': 1, 'Never-married': 0, 'Separated': 1, 'Widowed': 1, 'Married-spouse-absent': 2, 'Married-AF-spouse': 2}).astype(int)
    df['relationship'] = df['relationship'].map({'Wife': 1 , 'Own-child': 0 , 'Husband': 1, 'Not-in-family': 0, 'Other-relative': 0, 'Unmarried': 0}).astype(int)
    df['race'] = df['race'].map({'White': 1, 'Asian-Pac-Islander': 0, 'Amer-Indian-Eskimo': 0, 'Other': 0, 'Black': 0}).astype(int)
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0}).astype(int)
    # process hours
    df.loc[(df['hours'] <= 40), 'hours'] = 0
    df.loc[(df['hours'] > 40), 'hours'] = 1
    # process nationality
#     df.loc[(df['country'] != 'United-States'), 'country'] = 0
#     df.loc[(df['country'] == 'United-States'), 'country'] = 1
    df = df.drop(columns=['fnlwgt', 'education.num', 'occupation', 'country', 'capgain', 'caploss'])
#     df = df.drop(columns=['fnlwgt', 'education.num', 'occupation', 'capgain', 'caploss'])
    df = df.reset_index(drop=True)
    return df


def process_hmda(df):
    df = df[(df.action_taken==1) | (df.action_taken==3)]
    w_idx = df[(df['applicant_race-1']==5)
               &(pd.isna(df['applicant_race-2']))
               &(df['applicant_ethnicity-1']==2)].index
    b_idx = df[(df['applicant_race-1']==3)& (pd.isna(df['applicant_race-2']))].index
    df['race'] = -1
    df['race'].loc[w_idx] = 1
    df['race'].loc[b_idx] = 0
    df = df[df['race']>=0]
    df = df[df['debt_to_income_ratio']!='Exempt']
    df['gender'] = -1
    df['gender'][df['applicant_sex']==1] = 1
    df['gender'][df['applicant_sex']==2] = 0
    df = df[df['gender']>=0]

    df = df[['action_taken', 'income', 'race', 'gender', 'loan_type', 'applicant_age',
             'debt_to_income_ratio', 'loan_to_value_ratio', 'lien_status']]

    df['income'].fillna(71, inplace=True)
    df['loan_to_value_ratio'].fillna(93, inplace=True)
    df['debt_to_income_ratio'].fillna(41, inplace=True)

    df['applicant_age'] = df['applicant_age'].map({'25-34': 0, '35-44': 0, '<25': 0, '8888': -1,
                                                   '45-54': 1, '55-64': 1, '65-74': 1, '>74': 1}).astype(int)
    df = df[df['applicant_age']>=0]

    df['loan_to_value_ratio']= pd.to_numeric(df['loan_to_value_ratio'], errors= 'coerce')
    bins = np.array([40, 60, 79, 81, 90, 100])
    df['LV'] = np.ones_like(df['loan_to_value_ratio'])
    df['LV'][df['loan_to_value_ratio']<90] = 0
#     bins = np.array([0, 90, 100])
#     df['LV'] = np.digitize(df['loan_to_value_ratio'], bins)

    df.loc[df['debt_to_income_ratio']=='<20%', 'debt_to_income_ratio'] = 15
    df.loc[df['debt_to_income_ratio']=='20%-<30%', 'debt_to_income_ratio'] = 25
    df.loc[df['debt_to_income_ratio']=='30%-<36%', 'debt_to_income_ratio'] = 33
    df.loc[df['debt_to_income_ratio']=='50%-60%', 'debt_to_income_ratio'] = 55
    df.loc[df['debt_to_income_ratio']=='>60%', 'debt_to_income_ratio'] = 65
    df['debt_to_income_ratio'] = pd.to_numeric(df['debt_to_income_ratio'])
    bins = np.array([0, 20, 30, 36, 40, 45, 50, 60])
    bins = np.array([0, 30, 60, 90])
    df['DI'] = np.digitize(df['debt_to_income_ratio'], bins)

    bins = np.array([32, 53, 107, 374])
#     df['income_brackets'] = np.digitize(df['income'], bins)
    df['income_brackets'] = np.ones_like(df['income'])
    df['income_brackets'][df['income']<100] = 0

    df.dropna(inplace=True)
    df = df.reset_index(drop=True)
    df['action_taken'][df['action_taken']==3] = 0

    return df


def process_google(df):
    knn_imputer = KNNImputer(n_neighbors=5)
    df['Size'] = knn_imputer.fit_transform(df[['Size']])

    df.dropna(subset=['Content Rating', 'Install'], inplace=True)

    if df.isnull().sum().any():
        raise Exception("Data still contains missing values after preprocessing.")

    X = df.drop('Rating>4.2', axis=1)
    y = df['Rating>4.2']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Target Encoding for 'Category' and 'Genres'
    te = TargetEncoder()
    X_train['Category'] = te.fit_transform(X_train['Category'], y_train)
    X_test['Category'] = te.transform(X_test['Category'])
    X_train['Genres'] = te.fit_transform(X_train['Genres'], y_train)
    X_test['Genres'] = te.transform(X_test['Genres'])

    # One-Hot Encoding for 'Type' and 'Content Rating'
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    low_card_cols = ['Type', 'Content Rating']
    X_train_low_card = pd.DataFrame(ohe.fit_transform(X_train[low_card_cols]), index=X_train.index)
    X_test_low_card = pd.DataFrame(ohe.transform(X_test[low_card_cols]), index=X_test.index)
    X_train_low_card.columns = ohe.get_feature_names_out(low_card_cols)
    X_test_low_card.columns = ohe.get_feature_names_out(low_card_cols)

    # Combine all processed columns back into the training and testing dataframes
    X_train = pd.concat([X_train.drop(low_card_cols, axis=1), X_train_low_card], axis=1)
    X_test = pd.concat([X_test.drop(low_card_cols, axis=1), X_test_low_card], axis=1)

    return X_train, X_test, y_train, y_test
def process_bank_data(df):
    # Convert 'yes'/'no' to 1/0 for binary columns
    binary_cols = ['default', 'housing', 'loan', 'y']
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0}).astype(int)

    # Map education column to numerical values
    education_map = {
        'illiterate': 0,
        'primary': 1,
        'secondary': 2,
        'tertiary': 3,
        'unknown': 4
    }
    df['education'] = df['education'].map(education_map).astype(int)

    # Map job column to numerical values
    job_map = {
        'unemployed': 0,
        'housemaid': 1,
        'student': 2,
        'blue-collar': 3,
        'services': 4,
        'technician': 5,
        'admin.': 6,
        'self-employed': 7,
        'entrepreneur': 8,
        'retired': 9,
        'management': 10,
        'unknown': 11
    }
    df['job'] = df['job'].map(job_map).astype(int)

    # Map month column to numerical values
    month_map = {
        'jan': 1,
        'feb': 2,
        'mar': 3,
        'apr': 4,
        'may': 5,
        'jun': 6,
        'jul': 7,
        'aug': 8,
        'sep': 9,
        'oct': 10,
        'nov': 11,
        'dec': 12
    }
    df['month'] = df['month'].map(month_map).astype(int)

    # Map poutcome column to numerical values
    poutcome_map = {
        'failure': 0,
        'success': 1,
        'other': 2,
        'unknown': 3
    }
    df['poutcome'] = df['poutcome'].map(poutcome_map).astype(int)

    # Determine which columns to target encode or one-hot encode based on the number of unique values
    high_card_cols = []
    low_card_cols = []
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() > 5:
            high_card_cols.append(col)
        else:
            low_card_cols.append(col)

    # Target Encoding for high cardinality columns
    if high_card_cols:
        te = TargetEncoder()
        df[high_card_cols] = te.fit_transform(df[high_card_cols], df['y'])

    # One-Hot Encoding for low cardinality columns
    if low_card_cols:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_data = pd.DataFrame(ohe.fit_transform(df[low_card_cols]), index=df.index)

        # Handling different versions of OneHotEncoder
        try:
            encoded_data.columns = ohe.get_feature_names_out(low_card_cols)
        except AttributeError:
            encoded_data.columns = ohe.get_feature_names(low_card_cols)

        # Drop original low cardinality categorical variables and merge encoded data
        df = df.drop(columns=low_card_cols)
        df = pd.concat([df, encoded_data], axis=1)

    # Drop any remaining missing values
    df.dropna(inplace=True)

    return df

def process_student_scores(df):
    # Drop the 'Unnamed: 0' column
    df = df.drop(columns=['Unnamed: 0'])

    # Fill remaining missing values with 'unknown'
    df.fillna('unknown', inplace=True)

    # Mapping for ordered categorical variables
    parent_educ_map = {
        "bachelor's degree": 4,
        'some college': 3,
        "master's degree": 5,
        "associate's degree": 2,
        'high school': 1,
        'some high school': 0,
        'unknown': -1
    }
    lunch_type_map = {
        'free/reduced': 0,
        'standard': 1,
        'unknown': -1
    }
    practice_sport_map = {
        'never': 0,
        'sometimes': 1,
        'regularly': 2,
        'unknown': -1
    }
    wkly_study_hours_map = {
        '< 5': 0,
        '5 - 10': 1,
        '> 10': 2,
        'unknown': -1
    }
    gender_map = {
        'male': 1,
        'female': 0
    }
    test_prep_map = {
        'none': 0,
        'completed': 1,
        'unknown': -1
    }
    parent_marital_status_map = {
        'married': 0,
        'single': 1,
        'widowed': 2,
        'divorced': 3,
        'unknown': -1
    }
    is_first_child_map = {
        'yes': 1,
        'no': 0,
        'unknown': -1
    }
    transport_means_map = {
        'school_bus': 0,
        'private': 1,
        'unknown': -1
    }

    # Apply mappings
    df['ParentEduc'] = df['ParentEduc'].map(parent_educ_map).astype(int)
    df['LunchType'] = df['LunchType'].map(lunch_type_map).astype(int)
    df['PracticeSport'] = df['PracticeSport'].map(practice_sport_map).astype(int)
    df['WklyStudyHours'] = df['WklyStudyHours'].map(wkly_study_hours_map).astype(int)
    df['Gender'] = df['Gender'].map(gender_map).astype(int)
    df['TestPrep'] = df['TestPrep'].map(test_prep_map).astype(int)
    df['ParentMaritalStatus'] = df['ParentMaritalStatus'].map(parent_marital_status_map).astype(int)
    df['IsFirstChild'] = df['IsFirstChild'].map(is_first_child_map).astype(int)
    df['TransportMeans'] = df['TransportMeans'].map(transport_means_map).astype(int)

    # Remove remaining rows with 'unknown' in NrSiblings column
    df = df[df['NrSiblings'] != 'unknown']

    # Convert 'NrSiblings' to numeric, coercing errors
    df['NrSiblings'] = pd.to_numeric(df['NrSiblings'], errors='coerce')

    # Create binary target variable
    df['target'] = ((df['MathScore'] >= 60) & (df['ReadingScore'] >= 60) & (df['WritingScore'] >= 60)).astype(int)

    # Define categorical columns
    categorical_cols = ['EthnicGroup']

    # Separate columns into high and low cardinality
    high_card_cols = [col for col in categorical_cols if df[col].nunique() > 5]
    low_card_cols = [col for col in categorical_cols if df[col].nunique() <= 5]

    # Target Encoding for high cardinality columns
    te = TargetEncoder()
    df[high_card_cols] = te.fit_transform(df[high_card_cols], df['target'])

    # One-Hot Encoding for low cardinality columns
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_data = pd.DataFrame(ohe.fit_transform(df[low_card_cols]), index=df.index)

    # Handling different versions of OneHotEncoder
    try:
        encoded_data.columns = ohe.get_feature_names_out(low_card_cols)
    except AttributeError:
        encoded_data.columns = ohe.get_feature_names(low_card_cols)

    # Drop original low cardinality categorical variables and merge encoded data
    df = df.drop(columns=low_card_cols)
    df = pd.concat([df, encoded_data], axis=1)

    # Ensure all columns are numeric
    df = df.apply(pd.to_numeric, errors='ignore')

    # Separate features and target variable
    X = df.drop(columns=['MathScore', 'ReadingScore', 'WritingScore', 'target'])
    y = df['target']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def process_tfl(df):
    # Drop the 'date_of_incident' and 'year' columns
    df = df.drop(columns=['date_of_incident', 'year'])

    df = df[df['victims_age'] != 'Unknown']

    # Map 'victims_sex' column
    gender_map = {'Male': 1, 'Female': 0}
    df['victims_sex'] = df['victims_sex'].map(gender_map)

    # Remove rows with 'Unknown' in 'victims_sex'
    df = df.dropna(subset=['victims_sex'])

    # Map 'injury_result_description' to create binary target variable
    injury_map = {
        'Injuries treated on scene': 1,
        'Taken to Hospital – Reported Serious Injury or Severity Unknown': 1
    }

    df['injury_result_description'] = df['injury_result_description'].map(injury_map).fillna(0).astype(int)

    # Map 'victims_age'
    age_encoder = OrdinalEncoder(categories=[['Child', 'Youth', 'Adult', 'Elderly']])
    df['victims_age'] = age_encoder.fit_transform(df[['victims_age']])

    # Remove rows with 'Unknown' in 'victims_age'
    df = df.dropna(subset=['victims_age'])

    # Identify high and low cardinality columns
    high_card_cols = [col for col in df.select_dtypes(include=['object']).columns if df[col].nunique() > 5]
    low_card_cols = [col for col in df.select_dtypes(include=['object']).columns if df[col].nunique() <= 5]

    # Apply TargetEncoder for high cardinality columns
    if high_card_cols:
        te = TargetEncoder()
        df[high_card_cols] = te.fit_transform(df[high_card_cols], df['injury_result_description'])

    # Apply OneHotEncoder for low cardinality columns
    if low_card_cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = pd.DataFrame(ohe.fit_transform(df[low_card_cols]), index=df.index)

        # Handle column names for OneHotEncoder
        try:
            encoded_data.columns = ohe.get_feature_names_out(low_card_cols)
        except AttributeError:
            encoded_data.columns = ohe.get_feature_names(low_card_cols)

        df = df.drop(columns=low_card_cols)
        df = pd.concat([df, encoded_data], axis=1)

    # Drop rows with missing values
    df = df.dropna()

    # # Save the processed dataframe to a CSV file before splitting
    # df.to_csv('processed_tfl_data.csv', index=False)

    # Separate features and target variable
    X = df.drop(columns=['injury_result_description'])
    y = df['injury_result_description']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def process_insurance(df):

    df = df.drop(columns=['id'])

    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0}).astype(int)

    df['Vehicle_Damage'] = df['Vehicle_Damage'].map({'Yes': 1, 'No': 0}).astype(int)

    vehicle_age_map = {
        '< 1 Year': 0,
        '1-2 Year': 1,
        '> 2 Years': 2
    }
    df['Vehicle_Age'] = df['Vehicle_Age'].map(vehicle_age_map)

    df = df.dropna(subset=['Vehicle_Age'])

    df = df.dropna()

    X = df.drop(columns=['Response'])
    y = df['Response']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def process_cancer_data(df):
    # Drop the 'Patient_ID' column
    df = df.drop(columns=['Patient_ID'])

    # Map 'Gender' column
    gender_map = {'Male': 1, 'Female': 0}
    df['Gender'] = df['Gender'].map(gender_map).astype(int)

    # Map 'Stage' column
    stage_map = {'Stage IV': 1, 'Stage III': 1}
    df['Stage'] = df['Stage'].map(stage_map).fillna(0).astype(int)

    # Define ordered mappings
    ordered_mappings = {
        'Smoking_History': {'Never Smoked': 0, 'Former Smoker': 1, 'Current Smoker': 2},
        'Tumor_Location': {'Lower Lobe': 0, 'Middle Lobe': 1, 'Upper Lobe': 2}
    }

    # Apply mappings for ordered categorical variables
    for col, mapping in ordered_mappings.items():
        df[col] = df[col].map(mapping).astype(int)

    # Define binary mappings for Yes/No columns
    binary_mappings = {
        'Family_History': {'No': 0, 'Yes': 1},
        'Comorbidity_Diabetes': {'No': 0, 'Yes': 1},
        'Comorbidity_Hypertension': {'No': 0, 'Yes': 1},
        'Comorbidity_Heart_Disease': {'No': 0, 'Yes': 1},
        'Comorbidity_Chronic_Lung_Disease': {'No': 0, 'Yes': 1},
        'Comorbidity_Kidney_Disease': {'No': 0, 'Yes': 1},
        'Comorbidity_Autoimmune_Disease': {'No': 0, 'Yes': 1},
        'Comorbidity_Other': {'No': 0, 'Yes': 1}
    }

    # Apply mappings for binary categorical variables
    for col, mapping in binary_mappings.items():
        df[col] = df[col].map(mapping).astype(int)

    # Identify high and low cardinality columns
    high_card_cols = [col for col in df.select_dtypes(include=['object']).columns if df[col].nunique() > 5]
    low_card_cols = [col for col in df.select_dtypes(include=['object']).columns if df[col].nunique() <= 5]

    # Apply TargetEncoder for high cardinality columns
    if high_card_cols:
        te = TargetEncoder()
        df[high_card_cols] = te.fit_transform(df[high_card_cols], df['Stage'])

    # Apply OneHotEncoder for low cardinality columns
    if low_card_cols:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

        encoded_data = pd.DataFrame(ohe.fit_transform(df[low_card_cols]), index=df.index)

        # Handle column names for OneHotEncoder
        try:
            encoded_data.columns = ohe.get_feature_names_out(low_card_cols)
        except AttributeError:
            encoded_data.columns = ohe.get_feature_names(low_card_cols)

        df = df.drop(columns=low_card_cols)
        df = pd.concat([df, encoded_data], axis=1)

    # Drop rows with missing values
    df = df.dropna()

    # Separate features and target variable
    X = df.drop(columns=['Stage'])
    y = df['Stage']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def process_churn(df):
    # Drop the 'customerID' column
    df = df.drop(columns=['customerID', 'MonthlyCharges'])

    # Map 'gender' column to binary values
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0}).astype(int)

    # Map 'Churn' to binary target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).astype(int)

    # Identify binary columns and map Yes/No to 1/0
    binary_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype == 'object']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0}).astype(int)

    # Identify high and low cardinality columns for non-numeric columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    high_card_cols = [col for col in categorical_cols if df[col].nunique() > 5]
    low_card_cols = [col for col in categorical_cols if df[col].nunique() <= 5]

    # Apply TargetEncoder for high cardinality non-numeric columns
    if high_card_cols:
        te = TargetEncoder()
        df[high_card_cols] = te.fit_transform(df[high_card_cols], df['Churn'])

    # Apply OneHotEncoder for low cardinality non-numeric columns
    if low_card_cols:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_data = pd.DataFrame(ohe.fit_transform(df[low_card_cols]), index=df.index)

        try:
            encoded_data.columns = ohe.get_feature_names_out(low_card_cols)
        except AttributeError:
            encoded_data.columns = ohe.get_feature_names(low_card_cols)

        df = df.drop(columns=low_card_cols)
        df = pd.concat([df, encoded_data], axis=1)

    # Ensure all columns are numeric
    df = df.apply(pd.to_numeric, errors='ignore')

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def process_academic(df):
    df = df.drop(columns=['id'])

    df['Target'] = df['Target'].map({'Graduate': 1, 'Dropout': 0, 'Enrolled': 0}).astype(int)

    X = df.drop(columns=['Target'])
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def process_loan_data(df):
    # Drop the Loan_ID column
    df = df.drop(columns=['Loan_ID'])

    # Fill missing values with a placeholder
    df.fillna('unknown', inplace=True)

    # Map Loan_Status to binary values
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0}).astype(int)

    # Mapping for ordered categorical variables
    education_map = {
        'Graduate': 1,
        'Not Graduate': 0,
        'unknown': -1
    }
    property_area_map = {
        'Urban': 2,
        'Semiurban': 1,
        'Rural': 0,
        'unknown': -1
    }
    dependents_map = {
        '0': 0,
        '1': 1,
        '2': 2,
        '3+': 3,
        'unknown': -1
    }

    # Apply mappings
    df['Education'] = df['Education'].map(education_map).astype(int)
    df['Property_Area'] = df['Property_Area'].map(property_area_map).astype(int)
    df['Dependents'] = df['Dependents'].map(dependents_map).astype(int)
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0, 'unknown': -1}).astype(int)
    df['Married'] = df['Married'].map({'Yes': 1, 'No': 0, 'unknown': -1}).astype(int)
    df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0, 'unknown': -1}).astype(int)

    # Columns with more than 5 unique values - target encoding
    high_card_cols = [col for col in df.columns if df[col].nunique() > 5 and col != 'Loan_Status']
    if high_card_cols:
        te = TargetEncoder()
        df[high_card_cols] = te.fit_transform(df[high_card_cols], df['Loan_Status'])

    # Columns with 5 or fewer unique values - one-hot encoding
    low_card_cols = [col for col in df.columns if df[col].nunique() <= 5 and col != 'Loan_Status']
    if low_card_cols:
        # Ensure all values in low_card_cols are strings
        df[low_card_cols] = df[low_card_cols].astype(str)
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_data = pd.DataFrame(ohe.fit_transform(df[low_card_cols]), index=df.index)

        # Handling different versions of OneHotEncoder
        try:
            encoded_data.columns = ohe.get_feature_names_out(low_card_cols)
        except AttributeError:
            encoded_data.columns = ohe.get_feature_names(low_card_cols)

        # Drop original low cardinality categorical variables and merge encoded data
        df = df.drop(columns=low_card_cols)
        df = pd.concat([df, encoded_data], axis=1)

    # Create binary target variable
    df['target'] = df['Loan_Status']

    # Separate features and target variable
    X = df.drop(columns=['Loan_Status', 'target'])
    y = df['target']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def process_employee(df):
    # Drop rows with missing values
    df = df.dropna()

    # Mapping columns with order
    education_map = {
        'Bachelors': 0,
        'Masters': 1,
        'PHD': 2
    }
    df['Education'] = df['Education'].map(education_map).astype(int)

    # Map binary columns
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0}).astype(int)
    df['EverBenched'] = df['EverBenched'].map({'Yes': 1, 'No': 0}).astype(int)

    # Identify high and low cardinality columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    high_card_cols = [col for col in categorical_cols if df[col].nunique() > 5]
    low_card_cols = [col for col in categorical_cols if df[col].nunique() <= 5]

    # Apply TargetEncoder for high cardinality columns
    if high_card_cols:
        te = TargetEncoder()
        df[high_card_cols] = te.fit_transform(df[high_card_cols], df['LeaveOrNot'])

    # Apply OneHotEncoder for low cardinality columns
    if low_card_cols:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_data = pd.DataFrame(ohe.fit_transform(df[low_card_cols]), index=df.index)

        try:
            encoded_data.columns = ohe.get_feature_names_out(low_card_cols)
        except AttributeError:
            encoded_data.columns = ohe.get_feature_names(low_card_cols)

        df = df.drop(columns=low_card_cols)
        df = pd.concat([df, encoded_data], axis=1)

    X = df.drop(columns=['LeaveOrNot'])
    y = df['LeaveOrNot']
    X['City_New_Delhi'] = X['City_New Delhi']
    del X['City_New Delhi']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def process_ada(df):
    df.dropna(subset=['nativeCountry'], inplace=True)

    df['label'] = df['label'].replace(-1, 0)

    if df.isnull().sum().any():
        raise Exception("Data still contains missing values after preprocessing.")

    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0}).astype(int)
    education_map = {
        'Preschool': 0,
        '1st-4th': 1,
        '5th-6th': 2,
        '7th-8th': 3,
        '9th': 4,
        '10th': 5,
        '11th': 6,
        '12th': 7,
        'HS-grad': 8,
        'Some-college': 9,
        'Bachelors': 10,
        'Assoc-acdm': 11,
        'Assoc-voc': 12,
        'Masters': 13,
        'Prof-school': 14,
        'Doctorate': 15
    }
    df['education'] = df['education'].map(education_map).astype(int)

    workclass_map = {
        'Never-worked': 0,
        'Without-pay': 1,
        'State-gov': 2,
        'Local-gov': 3,
        'Federal-gov': 4,
        'Self-emp-inc': 5,
        'Self-emp-not-inc': 6,
        'Private': 7
    }
    df['workclass'] = df['workclass'].map(workclass_map).astype(int)

    relationship_map = {
        'Wife': 1,
        'Own-child': 0,
        'Husband': 1,
        'Not-in-family': 0,
        'Other-relative': 0,
        'Unmarried': 0
    }
    df['relationship'] = df['relationship'].map(relationship_map).astype(int)

    race_map = {
        'White': 1,
        'Asian-Pac-Islander': 0,
        'Amer-Indian-Eskimo': 0,
        'Other': 0,
        'Black': 0
    }
    df['race'] = df['race'].map(race_map).astype(int)

    df['capitalGain'] = pd.qcut(df['capitalGain'].rank(method='first'), 10, labels=False, duplicates='drop')

    X = df.drop(columns=['label'])
    y = df['label']

    high_card_cols = ['nativeCountry', 'occupation']
    te = TargetEncoder()
    X[high_card_cols] = te.fit_transform(X[high_card_cols], y)

    low_card_cols = ['maritalStatus']
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_data = pd.DataFrame(ohe.fit_transform(X[low_card_cols]), index=X.index)

    # Handling different versions of OneHotEncoder
    try:
        encoded_data.columns = ohe.get_feature_names_out(low_card_cols)
    except AttributeError:
        encoded_data.columns = ohe.get_feature_names(low_card_cols)

    X = X.drop(columns=low_card_cols)
    X = pd.concat([X, encoded_data], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def process_compas(df):
    df['age_cat'] = df['age_cat'].map({'Less than 25': 0, '25 - 45': 1, 'Greater than 45': 2}).astype(int)
    df['score_text'] = df['score_text'].map({'Low': 0, 'Medium': 1, 'High': 2}).astype(int)
    df['race'] = df['race'].map({'Other': 0, 'African-American': 0, 'Hispanic': 0, 'Native American': 0, 'Asian': 0, 'Caucasian': 1}).astype(int)
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0}).astype(int)

    df.loc[(df['priors_count'] <= 5), 'priors_count'] = 0
    df.loc[(df['priors_count'] > 5) & (df['priors_count'] <= 15), 'priors_count'] = 1
    df.loc[(df['priors_count'] > 15), 'priors_count'] = 2

    df.loc[(df['juv_fel_count'] == 0), 'juv_fel_count'] = 0
    df.loc[(df['juv_fel_count'] == 1), 'juv_fel_count'] = 1
    df.loc[(df['juv_fel_count'] > 1), 'juv_fel_count'] = 2

    df.loc[(df['juv_misd_count'] == 0), 'juv_misd_count'] = 0
    df.loc[(df['juv_misd_count'] == 1), 'juv_misd_count'] = 1
    df.loc[(df['juv_misd_count'] > 1), 'juv_misd_count'] = 2

    df.loc[(df['juv_other_count'] == 0), 'juv_other_count'] = 0
    df.loc[(df['juv_other_count'] == 1), 'juv_other_count'] = 1
    df.loc[(df['juv_other_count'] > 1), 'juv_other_count'] = 2
    return df


def process_heart(df):
    # Drop the 'id' column
    df = df.drop(columns=['id', 'dataset'])

    # Map 'sex' to binary values
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0}).astype(int)

    # Map 'num' to binary target variable
    df['num'] = df['num'].apply(lambda x: 1 if x != 0 else 0)

    # Define ordered mappings for categorical columns
    ordered_mappings = {
        'cp': {'typical angina': 0, 'atypical angina': 1, 'non-anginal': 2, 'asymptomatic': 3},
        'restecg': {'normal': 0, 'stt abnormality': 1, 'lv hypertrophy': 2},
        'slope': {'upsloping': 0, 'flat': 1, 'downsloping': 2},
        'thal': {'normal': 0, 'fixed defect': 1, 'reversible defect': 2}
    }

    # Apply mappings for ordered categorical variables
    for col, mapping in ordered_mappings.items():
        df[col] = df[col].map(mapping)

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Convert mapped columns to integer
    for col, mapping in ordered_mappings.items():
        df[col] = df[col].astype(int)

    # Separate columns into high and low cardinality
    categorical_cols = df.select_dtypes(include=['object']).columns
    high_card_cols = [col for col in categorical_cols if df[col].nunique() > 5]
    low_card_cols = [col for col in categorical_cols if df[col].nunique() <= 5]

    # Target Encoding for high cardinality columns
    if high_card_cols:
        te = TargetEncoder()
        df[high_card_cols] = te.fit_transform(df[high_card_cols], df['num'])

    # One-Hot Encoding for low cardinality columns
    if low_card_cols:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_data = pd.DataFrame(ohe.fit_transform(df[low_card_cols]), index=df.index)

        try:
            encoded_data.columns = ohe.get_feature_names_out(low_card_cols)
        except AttributeError:
            encoded_data.columns = ohe.get_feature_names(low_card_cols)

        df = df.drop(columns=low_card_cols)
        df = pd.concat([df, encoded_data], axis=1)

    # Separate features and target variable
    X = df.drop(columns=['num'])
    y = df['num']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
def process_credit_card(df):

    df = df.drop(columns=['ID'])

    df['PAY'] = df[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].mean(axis=1)

    df['BILL_AMT'] = df[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].mean(axis=1)

    df['PAY_AMT'] = df[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].mean(axis=1)

    df = df.drop(columns=['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                          'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                          'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'])

    df['SEX'] = df['SEX'].map({1: 0, 2: 1}).astype(int)  # Male: 0, Female: 1

    df['default payment next month'] = df['default payment next month'].astype(int)

    num_cols = df.select_dtypes(include=['float64']).columns
    df[num_cols] = df[num_cols].astype(int)

    X = df.drop(columns=['default payment next month'])
    y = df['default payment next month']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def process_hr(df):
    df = df.drop(columns=['StandardHours', 'EmployeeCount', 'Over18'])

    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0}).astype(int)

    df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0}).astype(int)

    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0}).astype(int)

    business_travel_map = {
        'Non-Travel': 0,
        'Travel_Rarely': 1,
        'Travel_Frequently': 2
    }
    df['BusinessTravel'] = df['BusinessTravel'].map(business_travel_map).astype(int)

    high_card_cols = [col for col in df.select_dtypes(include=['object']).columns if df[col].nunique() > 5]
    low_card_cols = [col for col in df.select_dtypes(include=['object']).columns if df[col].nunique() <= 5]

    if high_card_cols:
        te = TargetEncoder()
        df[high_card_cols] = te.fit_transform(df[high_card_cols], df['Attrition'])

    if low_card_cols:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_data = pd.DataFrame(ohe.fit_transform(df[low_card_cols]), index=df.index)

        try:
            encoded_data.columns = ohe.get_feature_names_out(low_card_cols)
        except AttributeError:
            encoded_data.columns = ohe.get_feature_names(low_card_cols)

        df = df.drop(columns=low_card_cols)
        df = pd.concat([df, encoded_data], axis=1)

    return df


def load_hr():
    df = pd.read_csv('hr.csv', engine='python')
    df = process_hr(df)

    y = df['Attrition']
    X = df.drop(columns=['Attrition'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
def load_german(preprocess=True):
    cols = ['status', 'duration', 'credit_hist', 'purpose', 'credit_amt', 'savings', 'employment',\
            'install_rate', 'personal_status', 'debtors', 'residence', 'property', 'age', 'install_plans',\
            'housing', 'num_credits', 'job', 'num_liable', 'telephone', 'foreign_worker', 'credit']
    df = pd.read_table('german.data', names=cols, sep=" ", index_col=False)
    df['credit'] = df['credit'].replace(2, 0) #1 = Good, 2= Bad credit risk
    y = df['credit']
    df = process_german(df, preprocess)
    if preprocess:
        df = df.drop(columns=['purpose', 'personal_status', 'housing', 'credit'])
    else:
        df = df.drop(columns=['personal_status', 'credit'])

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return X_train, X_test, y_train, y_test


def load_adult(sample=False):
    cols = ['age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital', 'occupation',\
            'relationship', 'race', 'gender', 'capgain', 'caploss', 'hours', 'country', 'income']
    df_train = pd.read_csv('adult.data', names=cols, sep=", ", engine='python')
    df_test = pd.read_csv('adult.test', names=cols, sep=", ", engine='python')

    df_train = process_adult(df_train)
    df_test = process_adult(df_test)

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    X_train = df_train.drop(columns='income')
    y_train = df_train['income']

    X_test = df_test.drop(columns='income')
    y_test = df_test['income']
    return X_train, X_test, y_train, y_test


def load_mozilla():
    df = pd.read_csv('mozilla4.csv')

    y = df['state']
    X = df[['start', 'end', 'event', 'size']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def load_google():
    df = pd.read_csv('google.csv')
    X_train, X_test, y_train, y_test = process_google(df)

    return X_train, X_test, y_train, y_test

def load_heart_disease():
    df = pd.read_csv('heart_disease.csv')
    X_train, X_test, y_train, y_test = process_heart_disease(df)
    return X_train, X_test, y_train, y_test

def load_ada():
    df = pd.read_csv('ada.csv')
    X_train, X_test, y_train, y_test = process_ada(df)

    return X_train, X_test, y_train, y_test

def load_academic():
    df = pd.read_csv('academic.csv')
    X_train, X_test, y_train, y_test = process_academic(df)

    return X_train, X_test, y_train, y_test
def load_loan():
    df = pd.read_csv('loan.csv')
    X_train, X_test, y_train, y_test = process_loan_data(df)

    return X_train, X_test, y_train, y_test

def load_scores():
    df = pd.read_csv('scores.csv')
    X_train, X_test, y_train, y_test = process_student_scores(df)

    return X_train, X_test, y_train, y_test

def load_churn():
    df = pd.read_csv('churn.csv')
    X_train, X_test, y_train, y_test = process_churn(df)

    return X_train, X_test, y_train, y_test

def load_compas():
    df = pd.read_csv('compas-scores-two-years.csv')
    df = df[['event', 'is_violent_recid', 'is_recid', 'priors_count', 'juv_other_count',\
             'juv_misd_count', 'juv_fel_count', 'race', 'age_cat', 'sex','score_text']]
    df = process_compas(df)

    y = df['is_recid']
    # y = df['is_violent_recid']
    df = df.drop(columns=['is_recid', 'is_violent_recid'])

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return X_train, X_test, y_train, y_test

def load_traffic():
    df = pd.read_csv('traffic_violations_cleaned.csv')
    y = df['search_outcome']
    df = df.drop(columns=['search_outcome'])
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return X_train, X_test, y_train, y_test


def load_sqf():
    df_train = pd.read_csv('sqf_train.csv')
    y_train = df_train['frisked']
    df_train['inout'] = df_train['inout_I']
    df_train['gender'] = df_train['sex_M']

    X_train = df_train.drop(columns=['frisked', 'inout_I', 'inout_O', 'sex_M', 'sex_F'])
    proxy = y_train + X_train.gender*2 - np.random.binomial(n=1, p=0.2, size=len(y_train))
    X_train['proxy'] = np.where(proxy>=1.5, 1, 0)

    df_test = pd.read_csv('sqf_test.csv')
    y_test = df_test['frisked']
    df_test['inout'] = df_test['inout_I']
    df_test['gender'] = df_test['sex_M']
    X_test = df_test.drop(columns=['frisked', 'inout_I', 'inout_O', 'sex_M', 'sex_F'])
    proxy = y_test + X_test.gender*2 - np.random.binomial(n=1, p=0.2, size=len(y_test))
    X_test['proxy'] = np.where(proxy>=1.5, 1, 0)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return X_train, X_test, y_train, y_test


def load_law():
    df = pd.read_csv('law_dataset.csv')
    y = df['pass_bar']
    df = df.drop(columns=['pass_bar'])
    df = process_law(df)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)
    X_train = copy.deepcopy(X_train).reset_index(drop=True)
    X_test = copy.deepcopy(X_test).reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return X_train, X_test, y_train, y_test

def load_bank():
    df = pd.read_csv('bank.csv', sep=';')
    df = process_bank_data(df)
    y = df['y']
    X = df.drop(columns=['y'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def load_cancer():
    df = pd.read_csv('cancer.csv')
    X_train, X_test, y_train, y_test = process_cancer_data(df)
    return X_train, X_test, y_train, y_test

def load_insurance():
    df = pd.read_csv('insurance.csv')
    X_train, X_test, y_train, y_test = process_insurance(df)
    return X_train, X_test, y_train, y_test

def load_tfl():
    df = pd.read_csv('tfl.csv')
    X_train, X_test, y_train, y_test = process_tfl(df)
    return X_train, X_test, y_train, y_test

def load_employee():
    df = pd.read_csv('employee.csv')
    X_train, X_test, y_train, y_test = process_employee(df)
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_test.reset_index(drop=True)

def load_credit_card():
    df = pd.read_csv('credit_card_default.csv')
    X_train, X_test, y_train, y_test = process_credit_card(df)
    return X_train, X_test, y_train, y_test

def load_heart():
    df = pd.read_csv('heart.csv')
    X_train, X_test, y_train, y_test = process_heart(df)
    return X_train, X_test, y_train, y_test
def load_hmda_ca():
    data = pd.read_csv('races_White-Black or African American_loan_purposes_1_year_2019.csv')
    state = 'CA'
    df = data[data['state_code']==state].reset_index(drop=True)
    df = process_hmda(df)
    y = df['action_taken']
    df = df.drop(columns=['action_taken', 'income', 'debt_to_income_ratio', 'loan_to_value_ratio'])
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)
    X_train = copy.deepcopy(X_train).reset_index(drop=True)
    X_test = copy.deepcopy(X_test).reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    del data
    return X_train, X_test, y_train, y_test


def load_hmda_la():
    data = pd.read_csv('hmda-la-19.csv')
    state = 'LA'
    df = data[data['state_code']==state].reset_index(drop=True)
    df = process_hmda(df)
    y = df['action_taken']
    df = df.drop(columns=['action_taken', 'income', 'debt_to_income_ratio', 'loan_to_value_ratio'])
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)
    X_train = copy.deepcopy(X_train).reset_index(drop=True)
    X_test = copy.deepcopy(X_test).reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    del data
    return X_train, X_test, y_train, y_test


def load(dataset, preprocess=True, row_num=10000, attr_num=30, sample=False):
    if dataset == 'compas':
        return load_compas()
    elif dataset == 'adult':
        return load_adult(sample=sample)
    elif dataset == 'german':
        return load_german(preprocess)
    elif dataset == 'traffic':
        return load_traffic()
    elif dataset == 'sqf':
        return load_sqf()
    elif dataset == 'law':
        return load_law()
    elif dataset == 'hmda':
        return load_hmda_la()
    elif dataset == 'mozilla':
        return load_mozilla()
    elif dataset == 'random':
        return generate_random_dataset(row_num, attr_num)
    elif dataset == 'google':
        return load_google()
    elif dataset == 'ada':
        return load_ada()
    elif dataset == 'bank':
        return load_bank()
    elif dataset == 'scores':
        return load_scores()
    elif dataset == 'loan':
        return load_loan()
    elif dataset == 'cancer':
        return load_cancer()
    elif dataset == 'tfl':
        return load_tfl()
    elif dataset == 'academic':
        return load_academic()
    elif dataset == 'employee':
        return load_employee()
    elif dataset == 'credit_card':
        return load_credit_card()
    elif dataset == 'churn':
        return load_churn()
    elif dataset == 'hr':
        return load_hr()
    elif dataset == 'insurance':
        return load_insurance()
    else:
        raise NotImplementedError


def generate_random_dataset(row_num, attr_num):
    cols_ls = list()
    for attr_idx in range(attr_num):
        col = np.random.binomial(n=1, p=0.5, size=(row_num, 1))
        cols_ls.append(col)
    X_mat = np.concatenate(cols_ls, axis=1)
    noise = np.random.binomial(n=2, p=0.03, size=(row_num, 1))
    random_coef = np.random.random(attr_num)

    X = pd.DataFrame(X_mat, columns=[f'A{attr_idx}' for attr_idx in range(attr_num)])
    y = pd.Series(np.where((np.dot(X_mat, random_coef.reshape(-1, 1))+noise)>attr_num*0.25, 1, 0).ravel(), name='foo')
    X['AA'] = np.random.binomial(n=1, p=0.5, size=(row_num, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test
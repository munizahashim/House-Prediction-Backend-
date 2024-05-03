import os
import uuid
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from django.conf import settings
import numpy as np
import pickle



with open(r'C:\Users\muniza.hashim\Desktop\senior\FYP\FYP progress\House Prices\Preprocessing Data\PredictHouse(Backend)\PredictHouse(Backend)\project_name\house_prediction\static\Linear_regressor_model.pkl', 'rb') as file:
    linear_model = pickle.load(file)
with open(r'C:\Users\muniza.hashim\Desktop\senior\FYP\FYP progress\House Prices\Preprocessing Data\PredictHouse(Backend)\PredictHouse(Backend)\project_name\house_prediction\static\DecisionTreeRegressor_model.pkl', 'rb') as file1:
    decision_tree_model = pickle.load(file1)
with open(r'C:\Users\muniza.hashim\Desktop\senior\FYP\FYP progress\House Prices\Preprocessing Data\PredictHouse(Backend)\PredictHouse(Backend)\project_name\house_prediction\static\RandomForestRegressor_model.pkl', 'rb') as file2:
    random_forest_model = pickle.load(file2)

# with open('CatBoostRegressor_model.pkl', 'rb') as file3:
#     catboost_model = pickle.load(file3)
with open(r'C:\Users\muniza.hashim\Desktop\senior\FYP\FYP progress\House Prices\Preprocessing Data\PredictHouse(Backend)\PredictHouse(Backend)\project_name\house_prediction\static\lgbm_model.pkl', 'rb') as file4:
    lgbm_model = pickle.load(file4)

# Load label encoders
with open(r'C:\Users\muniza.hashim\Desktop\senior\FYP\FYP progress\House Prices\Preprocessing Data\PredictHouse(Backend)\PredictHouse(Backend)\project_name\house_prediction\static\district_encoder.pkl', 'rb') as file5:
    district_encoder = pickle.load(file5)
with open(r'C:\Users\muniza.hashim\Desktop\senior\FYP\FYP progress\House Prices\Preprocessing Data\PredictHouse(Backend)\PredictHouse(Backend)\project_name\house_prediction\static\micro_district_encoder.pkl', 'rb') as file6:
    micro_district_encoder = pickle.load(file6)
with open(r'C:\Users\muniza.hashim\Desktop\senior\FYP\FYP progress\House Prices\Preprocessing Data\PredictHouse(Backend)\PredictHouse(Backend)\project_name\house_prediction\static\building_type_encoder.pkl', 'rb') as file7:
    building_type_encoder = pickle.load(file7)
with open(r'C:\Users\muniza.hashim\Desktop\senior\FYP\FYP progress\House Prices\Preprocessing Data\PredictHouse(Backend)\PredictHouse(Backend)\project_name\house_prediction\static\condition_encoder.pkl', 'rb') as file8:
    condition_encoder = pickle.load(file8)
def generate_correlation_matrix_image(house_data, independent_variables, dependent_variable):
    import matplotlib
    matplotlib.use('Agg')  # Use the Agg backend

    corr_matrix = house_data[independent_variables + [dependent_variable]].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')

    # Generate a random filename using UUID
    random_filename = str(uuid.uuid4()) + '.png'

    # Specify the full path to save the image in the media folder
    image_path = os.path.join(settings.MEDIA_ROOT, random_filename)

    # Save the image
    plt.savefig(image_path)
    plt.close()

    return image_path  # Return the image path


def generate_pairplot_image(house_data, independent_variables, dependent_variable):
    import matplotlib
    matplotlib.use('Agg')
    pairplot_data = house_data[independent_variables + [dependent_variable]].sample(min(500, len(house_data)), random_state=1)
    pairplot = sns.pairplot(pairplot_data)
    pairplot.set(title='Pairplot')

    # Generate a random filename using UUID
    random_filename = str(uuid.uuid4()) + '.png'

    # Specify the full path to save the image in the media folder
    image_path = os.path.join(settings.MEDIA_ROOT, random_filename)

    # Save the image
    plt.savefig(image_path)
    plt.close()

    return image_path  # Return the image path

def generate_distribution_plot_image(house_data, dependent_variable):
    import matplotlib
    matplotlib.use('Agg')  # Use the Agg backend

    plt.figure(figsize=(10, 8))
    sns.histplot(house_data[dependent_variable], bins=50, kde=True)
    plt.title('Price Distribution')

    # Generate a random filename using UUID
    random_filename = str(uuid.uuid4()) + '.png'

    # Specify the full path to save the image in the media folder
    image_path = os.path.join(settings.MEDIA_ROOT, random_filename)

    # Save the image
    plt.savefig(image_path)
    plt.close()

    return image_path  # Return the image path

def generate_trend_line_plot_image(house_data, dependent_variable):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the trend line
    ax.plot(house_data['date_year'], house_data[dependent_variable], marker='o', linestyle='-')

    # Set labels and title
    ax.set_xlabel('Date Year')
    ax.set_ylabel(dependent_variable)
    ax.set_title('Price Trends')

    # Generate a random filename using UUID
    random_filename = str(uuid.uuid4()) + '.png'

    # Specify the full path to save the image in the media folder
    image_path = os.path.join(settings.MEDIA_ROOT, random_filename)

    # Save the plot as a PNG image
    plt.savefig(image_path)

    # Close the plot to free up memory
    plt.close()

    return image_path

def generate_pie_chart_image(house_data, dependent_variable):
    # Calculate total price by district
    price_by_district = house_data.groupby('district')[dependent_variable].sum()

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the pie chart
    ax.pie(price_by_district, labels=price_by_district.index, autopct='%1.1f%%')

    # Set title
    ax.set_title('House Prices by District')

    # Generate a random filename using UUID
    random_filename = str(uuid.uuid4()) + '.png'

    # Specify the full path to save the image in the media folder
    image_path = os.path.join(settings.MEDIA_ROOT, random_filename)

    # Save the plot as a PNG image
    plt.savefig(image_path)

    # Close the plot to free up memory
    plt.close()

    return image_path  # Return the image path

def generate_scatter_plot_image(house_data, independent_var_choice, dependent_variable):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the scatter plot
    ax.scatter(house_data[independent_var_choice], house_data[dependent_variable])

    # Set labels and title
    ax.set_xlabel(independent_var_choice)
    ax.set_ylabel(dependent_variable)
    ax.set_title(f'House Prices vs {independent_var_choice}')

    # Generate a random filename using UUID
    random_filename = str(uuid.uuid4()) + '.png'

    # Specify the full path to save the image in the media folder
    image_path = os.path.join(settings.MEDIA_ROOT, random_filename)

    # Save the plot as a PNG image
    plt.savefig(image_path)

    # Close the plot to free up memory
    plt.close()
    return image_path

def predict(formData):
    try:
        selected_model = formData["model"]
        district_encoded = district_encoder.transform([formData["district"]])[0]
        micro_district_encoded = micro_district_encoder.transform([formData["microdistrict"]])[0]
        building_type_encoded = building_type_encoder.transform([formData["buildingType"]])[0]
        condition_encoded = condition_encoder.transform([formData["condition"]])[0]
        input_data = np.array([
            float(formData["squareMeters"]),  # Assuming this corresponds to "square" in your original code
            int(formData["numOfRooms"]),      # Assuming this corresponds to "rooms" in your original code
            int(formData["numOfFloors"]),     # Assuming this corresponds to "floors" in your original code
            int(formData["floorNumber"]),     # Assuming this corresponds to "floor" in your original code
            int(formData["yearOfConstruction"]),  # Assuming this corresponds to "date_year" in your original code
            district_encoded,  # Assuming this corresponds to "district_encoded" in your original code
            micro_district_encoded,
            building_type_encoded,
            condition_encoded,
        ])
        if selected_model == 'Linear Regression':
            prediction = linear_model.predict(input_data.reshape(1, -1))[0]
            model_summary_data = {
                "Model": "Linear Regression",
                "MSE": 489020200.79406863,
                "R Squared": 0.7507187181330213
            }
        elif selected_model == 'Decision Tree':
            prediction = decision_tree_model.predict(input_data.reshape(1, -1))[0]
            model_summary_data = {
                "Model": "Decision Tree Regressor",
                "MSE": 574342995.740365,
                "R Squared": 0.7354839965380982
            }
        elif selected_model == 'Random Forest':
            prediction = random_forest_model.predict(input_data.reshape(1, -1))[0]
            model_summary_data = {
                "Model": "Random Forest Regressor",
                "MSE": 192164806.4368714,
                "R Squared": 0.9020427188886709
            }
        elif selected_model == 'LGBM':
            prediction = lgbm_model.predict(input_data.reshape(1, -1))[0]
            model_summary_data = {
                "Model": "LGBM",
                "MSE": 370814781.9969399,
                "R Squared": 0.8377896010239424
            }
        elif selected_model == 'CATBOOST':
            prediction = lgbm_model.predict(input_data.reshape(1, -1))[0]
            model_summary_data = {
                "Model": "CATBOOST",
                "MSE": 360614704.47670025,
                "R Squared": 0.8422515554132322
            }
        return {"prediction": prediction, "model_summary_data": model_summary_data}
    except KeyError as e:
        return {"error": str(e)}
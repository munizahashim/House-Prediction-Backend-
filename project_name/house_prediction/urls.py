
from django.urls import path
from .views import detailed_analysis, analyze_price, predict_price  # Import the view function from views.py

urlpatterns = [
    path('detailed-analysis/', detailed_analysis, name='graph_data'),
    path("analyze-price/", analyze_price, name="price-analysis"),# Define the URL pattern
    path("predict-price/", predict_price, name="predict-price")
]

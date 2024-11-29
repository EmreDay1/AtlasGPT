import pandas as pd
import random
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, pipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import re
from geopy.geocoders import Nominatim
from meteostat import Point, Daily
from datetime import datetime
from geopy.geocoders import Nominatim

# Example weather prompts
weather_prompts = [
    "Show the temperature in New York, USA from 2021, 01, 01 to 2021, 01, 31",
    "Display the precipitation in Paris, France from 2022, 05, 01 to 2022, 05, 31",
    "Highlight the snowfall in Tokyo, Japan from 2020, 12, 01 to 2020, 12, 31",
    "Show the wind speed in Sydney, Australia from 2019, 06, 01 to 2019, 06, 30",
    "Display the wind gust in Berlin, Germany from 2021, 03, 01 to 2021, 03, 31",
    "Highlight the wind direction in Rome, Italy from 2021, 07, 01 to 2021, 07, 31",
    "Show the pressure in Moscow, Russia from 2020, 11, 01 to 2020, 11, 30",
    "Display the sunshine duration in Toronto, Canada from 2021, 02, 01 to 2021, 02, 28",
    "Show the temperature in Beijing, China from 2020, 08, 01 to 2020, 08, 31",
    "Display the precipitation in Mumbai, India from 2021, 09, 01 to 2021, 09, 30",
    "Highlight the snowfall in Oslo, Norway from 2021, 01, 01 to 2021, 01, 31",
    "Show the wind speed in Cairo, Egypt from 2020, 05, 01 to 2020, 05, 31",
    "Display the wind gust in Buenos Aires, Argentina from 2021, 11, 01 to 2021, 11, 30",
    "Highlight the wind direction in Nairobi, Kenya from 2021, 04, 01 to 2021, 04, 30",
    "Show the pressure in Madrid, Spain from 2020, 07, 01 to 2020, 07, 31",
    "Display the sunshine duration in Bangkok, Thailand from 2021, 12, 01 to 2021, 12, 31",
    "Show the temperature in London, UK from 2022, 01, 01 to 2022, 01, 31",
    "Display the precipitation in Istanbul, Turkey from 2020, 03, 01 to 2020, 03, 31",
    "Highlight the snowfall in Seoul, South Korea from 2021, 02, 01 to 2021, 02, 28",
    "Show the wind speed in Mexico City, Mexico from 2021, 06, 01 to 2021, 06, 30",
    "Display the wind gust in São Paulo, Brazil from 2020, 09, 01 to 2020, 09, 30",
    "Highlight the wind direction in Johannesburg, South Africa from 2021, 10, 01 to 2021, 10, 31",
    "Show the pressure in Athens, Greece from 2020, 04, 01 to 2020, 04, 30",
    "Display the sunshine duration in Lisbon, Portugal from 2021, 08, 01 to 2021, 08, 31",
    "Show the temperature in Sydney, Australia from 2021, 01, 01 to 2021, 01, 31",
    "Display the precipitation in Moscow, Russia from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the snowfall in Toronto, Canada from 2021, 12, 01 to 2021, 12, 31",
    "Show the wind speed in Beijing, China from 2020, 10, 01 to 2020, 10, 31",
    "Display the wind gust in Mumbai, India from 2021, 08, 01 to 2021, 08, 31",
    "Highlight the wind direction in New York, USA from 2021, 02, 01 to 2021, 02, 28",
    "Show the pressure in Paris, France from 2020, 05, 01 to 2020, 05, 31",
    "Display the sunshine duration in Tokyo, Japan from 2021, 03, 01 to 2021, 03, 31",
    "Show the temperature in Berlin, Germany from 2020, 07, 01 to 2020, 07, 31",
    "Display the precipitation in Rome, Italy from 2021, 09, 01 to 2021, 09, 30",
    "Highlight the snowfall in Oslo, Norway from 2020, 01, 01 to 2020, 01, 31",
    "Show the wind speed in Cairo, Egypt from 2021, 06, 01 to 2021, 06, 30",
    "Display the wind gust in Buenos Aires, Argentina from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the wind direction in Nairobi, Kenya from 2021, 12, 01 to 2021, 12, 31",
    "Show the pressure in Madrid, Spain from 2020, 04, 01 to 2020, 04, 30",
    "Display the sunshine duration in Bangkok, Thailand from 2021, 08, 01 to 2021, 08, 31",
    "Show the temperature in London, UK from 2022, 06, 01 to 2022, 06, 30",
    "Display the precipitation in Istanbul, Turkey from 2020, 02, 01 to 2020, 02, 29",
    "Highlight the snowfall in Seoul, South Korea from 2021, 05, 01 to 2021, 05, 31",
    "Show the wind speed in Mexico City, Mexico from 2021, 03, 01 to 2021, 03, 31",
    "Display the wind gust in São Paulo, Brazil from 2020, 08, 01 to 2020, 08, 31",
    "Highlight the wind direction in Johannesburg, South Africa from 2021, 04, 01 to 2021, 04, 30",
    "Show the pressure in Athens, Greece from 2020, 10, 01 to 2020, 10, 31",
    "Display the sunshine duration in Lisbon, Portugal from 2021, 09, 01 to 2021, 09, 30",
    "Show the temperature in Sydney, Australia from 2021, 07, 01 to 2021, 07, 31",
    "Display the precipitation in Moscow, Russia from 2020, 12, 01 to 2020, 12, 31",
    "Highlight the snowfall in Toronto, Canada from 2021, 11, 01 to 2021, 11, 30",
    "Show the wind speed in Beijing, China from 2020, 06, 01 to 2020, 06, 30",
    "Display the wind gust in Mumbai, India from 2021, 05, 01 to 2021, 05, 31",
    "Highlight the wind direction in New York, USA from 2021, 03, 01 to 2021, 03, 31",
    "Show the pressure in Paris, France from 2020, 09, 01 to 2020, 09, 30",
    "Display the sunshine duration in Tokyo, Japan from 2021, 06, 01 to 2021, 06, 30",
    "Show the temperature in Berlin, Germany from 2020, 08, 01 to 2020, 08, 31",
    "Display the precipitation in Rome, Italy from 2021, 04, 01 to 2021, 04, 30",
    "Highlight the snowfall in Oslo, Norway from 2020, 02, 01 to 2020, 02, 29",
    "Show the wind speed in Cairo, Egypt from 2021, 08, 01 to 2021, 08, 31",
    "Display the wind gust in Buenos Aires, Argentina from 2020, 05, 01 to 2020, 05, 31",
    "Highlight the wind direction in Nairobi, Kenya from 2021, 10, 01 to 2021, 10, 31",
    "Show the pressure in Madrid, Spain from 2020, 03, 01 to 2020, 03, 31",
    "Display the sunshine duration in Bangkok, Thailand from 2021, 11, 01 to 2021, 11, 30",
    "Show the temperature in London, UK from 2022, 04, 01 to 2022, 04, 30",
    "Display the precipitation in Istanbul, Turkey from 2020, 07, 01 to 2020, 07, 31",
    "Highlight the snowfall in Seoul, South Korea from 2021, 08, 01 to 2021, 08, 31",
    "Show the wind speed in Mexico City, Mexico from 2021, 02, 01 to 2021, 02, 28",
    "Display the wind gust in São Paulo, Brazil from 2020, 12, 01 to 2020, 12, 31",
    "Highlight the wind direction in Johannesburg, South Africa from 2021, 01, 01 to 2021, 01, 31",
    "Show the pressure in Athens, Greece from 2020, 11, 01 to 2020, 11, 30",
    "Display the sunshine duration in Lisbon, Portugal from 2021, 12, 01 to 2021, 12, 31",
    "Show the temperature in Los Angeles, USA from 2020, 01, 01 to 2020, 01, 31",
    "Display the precipitation in Rio de Janeiro, Brazil from 2021, 01, 01 to 2021, 01, 31",
    "Highlight the snowfall in Zurich, Switzerland from 2020, 12, 01 to 2020, 12, 31",
    "Show the wind speed in Manila, Philippines from 2019, 06, 01 to 2019, 06, 30",
    "Display the wind gust in Madrid, Spain from 2021, 03, 01 to 2021, 03, 31",
    "Highlight the wind direction in Dublin, Ireland from 2021, 07, 01 to 2021, 07, 31",
    "Show the pressure in Vienna, Austria from 2020, 11, 01 to 2020, 11, 30",
    "Display the sunshine duration in Vancouver, Canada from 2021, 02, 01 to 2021, 02, 28",
    "Show the temperature in Seoul, South Korea from 2020, 08, 01 to 2020, 08, 31",
    "Display the precipitation in Delhi, India from 2021, 09, 01 to 2021, 09, 30",
    "Highlight the snowfall in Helsinki, Finland from 2021, 01, 01 to 2021, 01, 31",
    "Show the wind speed in Buenos Aires, Argentina from 2020, 05, 01 to 2020, 05, 31",
    "Display the wind gust in Nairobi, Kenya from 2021, 11, 01 to 2021, 11, 30",
    "Highlight the wind direction in Cairo, Egypt from 2021, 04, 01 to 2021, 04, 30",
    "Show the pressure in Barcelona, Spain from 2020, 07, 01 to 2020, 07, 31",
    "Display the sunshine duration in Kuala Lumpur, Malaysia from 2021, 12, 01 to 2021, 12, 31",
    "Show the temperature in Milan, Italy from 2022, 01, 01 to 2022, 01, 31",
    "Display the precipitation in Stockholm, Sweden from 2020, 03, 01 to 2020, 03, 31",
    "Highlight the snowfall in Oslo, Norway from 2021, 02, 01 to 2021, 02, 28",
    "Show the wind speed in Cape Town, South Africa from 2021, 06, 01 to 2021, 06, 30",
    "Display the wind gust in Santiago, Chile from 2020, 09, 01 to 2020, 09, 30",
    "Highlight the wind direction in Tehran, Iran from 2021, 10, 01 to 2021, 10, 31",
    "Show the pressure in Baghdad, Iraq from 2020, 04, 01 to 2020, 04, 30",
    "Display the sunshine duration in Melbourne, Australia from 2021, 08, 01 to 2021, 08, 31",
    "Show the temperature in Buenos Aires, Argentina from 2021, 01, 01 to 2021, 01, 31",
    "Display the precipitation in Cape Town, South Africa from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the snowfall in Tokyo, Japan from 2021, 12, 01 to 2021, 12, 31",
    "Show the wind speed in Rio de Janeiro, Brazil from 2020, 10, 01 to 2020, 10, 31",
    "Display the wind gust in Lima, Peru from 2021, 08, 01 to 2021, 08, 31",
    "Highlight the wind direction in Bogota, Colombia from 2021, 02, 01 to 2021, 02, 28",
    "Show the pressure in Santiago, Chile from 2020, 05, 01 to 2020, 05, 31",
    "Display the sunshine duration in Caracas, Venezuela from 2021, 03, 01 to 2021, 03, 31",
    "Show the temperature in Lima, Peru from 2020, 07, 01 to 2020, 07, 31",
    "Display the precipitation in Bogota, Colombia from 2021, 09, 01 to 2021, 09, 30",
    "Highlight the snowfall in La Paz, Bolivia from 2020, 01, 01 to 2020, 01, 31",
    "Show the wind speed in Quito, Ecuador from 2021, 06, 01 to 2021, 06, 30",
    "Display the wind gust in Asuncion, Paraguay from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the wind direction in Montevideo, Uruguay from 2021, 12, 01 to 2021, 12, 31",
    "Show the pressure in Georgetown, Guyana from 2020, 04, 01 to 2020, 04, 30",
    "Display the sunshine duration in Paramaribo, Suriname from 2021, 08, 01 to 2021, 08, 31",
    "Show the temperature in Johannesburg, South Africa from 2022, 01, 01 to 2022, 01, 31",
    "Display the precipitation in Nairobi, Kenya from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the snowfall in Algiers, Algeria from 2021, 12, 01 to 2021, 12, 31",
    "Show the wind speed in Accra, Ghana from 2020, 10, 01 to 2020, 10, 31",
    "Display the wind gust in Lagos, Nigeria from 2021, 08, 01 to 2021, 08, 31",
    "Highlight the wind direction in Dakar, Senegal from 2021, 02, 01 to 2021, 02, 28",
    "Show the pressure in Addis Ababa, Ethiopia from 2020, 05, 01 to 2020, 05, 31",
    "Display the sunshine duration in Khartoum, Sudan from 2021, 03, 01 to 2021, 03, 31",
    "Show the temperature in Casablanca, Morocco from 2020, 07, 01 to 2020, 07, 31",
    "Display the precipitation in Rabat, Morocco from 2021, 09, 01 to 2021, 09, 30",
    "Highlight the snowfall in Tunis, Tunisia from 2020, 01, 01 to 2020, 01, 31",
    "Show the wind speed in Tripoli, Libya from 2021, 06, 01 to 2021, 06, 30",
    "Display the wind gust in Cairo, Egypt from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the wind direction in Kampala, Uganda from 2021, 12, 01 to 2021, 12, 31",
    "Show the pressure in Kinshasa, Congo from 2020, 04, 01 to 2020, 04, 30",
    "Display the sunshine duration in Luanda, Angola from 2021, 08, 01 to 2021, 08, 31",
    "Show the temperature in Harare, Zimbabwe from 2022, 01, 01 to 2022, 01, 31",
    "Display the precipitation in Lusaka, Zambia from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the snowfall in Windhoek, Namibia from 2021, 12, 01 to 2021, 12, 31",
    "Show the wind speed in Gaborone, Botswana from 2020, 10, 01 to 2020, 10, 31",
    "Display the wind gust in Maputo, Mozambique from 2021, 08, 01 to 2021, 08, 31",
    "Highlight the wind direction in Antananarivo, Madagascar from 2021, 02, 01 to 2021, 02, 28",
    "Show the pressure in Port Louis, Mauritius from 2020, 05, 01 to 2020, 05, 31",
    "Display the sunshine duration in Moroni, Comoros from 2021, 03, 01 to 2021, 03, 31",
    "Show the temperature in Mogadishu, Somalia from 2020, 07, 01 to 2020, 07, 31",
    "Display the precipitation in Djibouti, Djibouti from 2021, 09, 01 to 2021, 09, 30",
    "Highlight the snowfall in Addis Ababa, Ethiopia from 2020, 01, 01 to 2020, 01, 31",
    "Show the wind speed in Nairobi, Kenya from 2021, 06, 01 to 2021, 06, 30",
    "Display the wind gust in Lusaka, Zambia from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the wind direction in Gaborone, Botswana from 2021, 12, 01 to 2021, 12, 31",
    "Show the pressure in Harare, Zimbabwe from 2020, 04, 01 to 2020, 04, 30",
    "Display the sunshine duration in Windhoek, Namibia from 2021, 08, 01 to 2021, 08, 31",
    "Show the temperature in Maputo, Mozambique from 2022, 01, 01 to 2022, 01, 31",
    "Display the precipitation in Antananarivo, Madagascar from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the snowfall in Port Louis, Mauritius from 2021, 12, 01 to 2021, 12, 31",
    "Show the wind speed in Moroni, Comoros from 2020, 10, 01 to 2020, 10, 31",
    "Display the wind gust in Mogadishu, Somalia from 2021, 08, 01 to 2021, 08, 31",
    "Highlight the wind direction in Djibouti, Djibouti from 2021, 02, 01 to 2021, 02, 28",
    "Show the pressure in Addis Ababa, Ethiopia from 2020, 05, 01 to 2020, 05, 31",
    "Display the sunshine duration in Nairobi, Kenya from 2021, 03, 01 to 2021, 03, 31",
    "Show the temperature in Lusaka, Zambia from 2020, 07, 01 to 2020, 07, 31",
    "Display the precipitation in Gaborone, Botswana from 2021, 09, 01 to 2021, 09, 30",
    "Highlight the snowfall in Harare, Zimbabwe from 2020, 01, 01 to 2020, 01, 31",
    "Show the wind speed in Windhoek, Namibia from 2021, 06, 01 to 2021, 06, 30",
    "Display the wind gust in Maputo, Mozambique from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the wind direction in Antananarivo, Madagascar from 2021, 12, 01 to 2021, 12, 31",
    "Show the pressure in Port Louis, Mauritius from 2020, 04, 01 to 2020, 04, 30",
    "Display the sunshine duration in Moroni, Comoros from 2021, 08, 01 to 2021, 08, 31",
    "Show the temperature in Mogadishu, Somalia from 2022, 01, 01 to 2022, 01, 31",
    "Display the precipitation in Djibouti, Djibouti from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the snowfall in Addis Ababa, Ethiopia from 2021, 12, 01 to 2021, 12, 31",
    "Show the wind speed in Nairobi, Kenya from 2020, 10, 01 to 2020, 10, 31",
    "Display the wind gust in Lusaka, Zambia from 2021, 08, 01 to 2021, 08, 31",
    "Highlight the wind direction in Gaborone, Botswana from 2021, 02, 01 to 2021, 02, 28",
    "Show the pressure in Harare, Zimbabwe from 2020, 05, 01 to 2020, 05, 31",
    "Display the sunshine duration in Windhoek, Namibia from 2021, 03, 01 to 2021, 03, 31",
    "Show the temperature in Maputo, Mozambique from 2020, 07, 01 to 2020, 07, 31",
    "Display the precipitation in Antananarivo, Madagascar from 2021, 09, 01 to 2021, 09, 30",
    "Highlight the snowfall in Port Louis, Mauritius from 2020, 01, 01 to 2020, 01, 31",
    "Show the wind speed in Moroni, Comoros from 2021, 06, 01 to 2021, 06, 30",
    "Display the wind gust in Mogadishu, Somalia from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the wind direction in Djibouti, Djibouti from 2021, 12, 01 to 2021, 12, 31",
    "Show the pressure in Addis Ababa, Ethiopia from 2020, 04, 01 to 2020, 04, 30",
    "Display the sunshine duration in Nairobi, Kenya from 2021, 08, 01 to 2021, 08, 31",
    "Show the temperature in Lusaka, Zambia from 2022, 01, 01 to 2022, 01, 31",
    "Display the precipitation in Gaborone, Botswana from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the snowfall in Harare, Zimbabwe from 2021, 12, 01 to 2021, 12, 31",
    "Show the wind speed in Windhoek, Namibia from 2020, 10, 01 to 2020, 10, 31",
    "Display the wind gust in Maputo, Mozambique from 2021, 08, 01 to 2021, 08, 31",
    "Highlight the wind direction in Antananarivo, Madagascar from 2021, 02, 01 to 2021, 02, 28",
    "Show the pressure in Port Louis, Mauritius from 2020, 05, 01 to 2020, 05, 31",
    "Display the sunshine duration in Moroni, Comoros from 2021, 03, 01 to 2021, 03, 31",
    "Show the temperature in Mogadishu, Somalia from 2020, 07, 01 to 2020, 07, 31",
    "Display the precipitation in Djibouti, Djibouti from 2021, 09, 01 to 2021, 09, 30",
    "Highlight the snowfall in Addis Ababa, Ethiopia from 2020, 01, 01 to 2020, 01, 31",
    "Show the wind speed in Nairobi, Kenya from 2021, 06, 01 to 2021, 06, 30",
    "Display the wind gust in Lusaka, Zambia from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the wind direction in Gaborone, Botswana from 2021, 12, 01 to 2021, 12, 31",
    "Show the pressure in Harare, Zimbabwe from 2020, 04, 01 to 2020, 04, 30",
    "Display the sunshine duration in Windhoek, Namibia from 2021, 08, 01 to 2021, 08, 31",
    "Show the temperature in Maputo, Mozambique from 2022, 01, 01 to 2022, 01, 31",
    "Display the precipitation in Antananarivo, Madagascar from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the snowfall in Port Louis, Mauritius from 2021, 12, 01 to 2021, 12, 31",
    "Show the wind speed in Moroni, Comoros from 2020, 10, 01 to 2020, 10, 31",
    "Display the wind gust in Mogadishu, Somalia from 2021, 08, 01 to 2021, 08, 31",
    "Highlight the wind direction in Djibouti, Djibouti from 2021, 02, 01 to 2021, 02, 28",
    "Show the pressure in Addis Ababa, Ethiopia from 2020, 05, 01 to 2020, 05, 31",
    "Display the sunshine duration in Nairobi, Kenya from 2021, 03, 01 to 2021, 03, 31",
    "Show the temperature in Lusaka, Zambia from 2020, 07, 01 to 2020, 07, 31",
    "Display the precipitation in Gaborone, Botswana from 2021, 09, 01 to 2021, 09, 30",
    "Highlight the snowfall in Harare, Zimbabwe from 2020, 01, 01 to 2020, 01, 31",
    "Show the wind speed in Windhoek, Namibia from 2021, 06, 01 to 2021, 06, 30",
    "Display the wind gust in Maputo, Mozambique from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the wind direction in Antananarivo, Madagascar from 2021, 12, 01 to 2021, 12, 31",
    "Show the pressure in Port Louis, Mauritius from 2020, 04, 01 to 2020, 04, 30",
    "Display the sunshine duration in Moroni, Comoros from 2021, 08, 01 to 2021, 08, 31",
    "Show the temperature in Mogadishu, Somalia from 2022, 01, 01 to 2022, 01, 31",
    "Display the precipitation in Djibouti, Djibouti from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the snowfall in Addis Ababa, Ethiopia from 2021, 12, 01 to 2021, 12, 31",
    "Show the wind speed in Nairobi, Kenya from 2020, 10, 01 to 2020, 10, 31",
    "Display the wind gust in Lusaka, Zambia from 2021, 08, 01 to 2021, 08, 31",
    "Highlight the wind direction in Gaborone, Botswana from 2021, 02, 01 to 2021, 02, 28",
    "Show the pressure in Harare, Zimbabwe from 2020, 05, 01 to 2020, 05, 31",
    "Display the sunshine duration in Windhoek, Namibia from 2021, 03, 01 to 2021, 03, 31",
    "Show the temperature in Maputo, Mozambique from 2020, 07, 01 to 2020, 07, 31",
    "Display the precipitation in Antananarivo, Madagascar from 2021, 09, 01 to 2021, 09, 30",
    "Highlight the snowfall in Port Louis, Mauritius from 2020, 01, 01 to 2020, 01, 31",
    "Show the wind speed in Moroni, Comoros from 2021, 06, 01 to 2021, 06, 30",
    "Display the wind gust in Mogadishu, Somalia from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the wind direction in Djibouti, Djibouti from 2021, 12, 01 to 2021, 12, 31",
    "Show the pressure in Addis Ababa, Ethiopia from 2020, 04, 01 to 2020, 04, 30",
    "Display the sunshine duration in Nairobi, Kenya from 2021, 08, 01 to 2021, 08, 31",
    "Show the temperature in Lusaka, Zambia from 2022, 01, 01 to 2022, 01, 31",
    "Display the precipitation in Gaborone, Botswana from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the snowfall in Harare, Zimbabwe from 2021, 12, 01 to 2021, 12, 31",
    "Show the wind speed in Windhoek, Namibia from 2020, 10, 01 to 2020, 10, 31",
    "Display the wind gust in Maputo, Mozambique from 2021, 08, 01 to 2021, 08, 31",
    "Highlight the wind direction in Antananarivo, Madagascar from 2021, 02, 01 to 2021, 02, 28",
    "Show the pressure in Port Louis, Mauritius from 2020, 05, 01 to 2020, 05, 31",
    "Display the sunshine duration in Moroni, Comoros from 2021, 03, 01 to 2021, 03, 31",
    "Show the temperature in Mogadishu, Somalia from 2020, 07, 01 to 2020, 07, 31",
    "Display the precipitation in Djibouti, Djibouti from 2021, 09, 01 to 2021, 09, 30",
    "Highlight the snowfall in Addis Ababa, Ethiopia from 2020, 01, 01 to 2020, 01, 31",
    "Show the wind speed in Nairobi, Kenya from 2021, 06, 01 to 2021, 06, 30",
    "Display the wind gust in Lusaka, Zambia from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the wind direction in Gaborone, Botswana from 2021, 12, 01 to 2021, 12, 31",
    "Show the pressure in Harare, Zimbabwe from 2020, 04, 01 to 2020, 04, 30",
    "Display the sunshine duration in Windhoek, Namibia from 2021, 08, 01 to 2021, 08, 31",
    "Show the temperature in Maputo, Mozambique from 2022, 01, 01 to 2022, 01, 31",
    "Display the precipitation in Antananarivo, Madagascar from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the snowfall in Port Louis, Mauritius from 2021, 12, 01 to 2021, 12, 31",
    "Show the wind speed in Moroni, Comoros from 2020, 10, 01 to 2020, 10, 31",
    "Display the wind gust in Mogadishu, Somalia from 2021, 08, 01 to 2021, 08, 31",
    "Highlight the wind direction in Djibouti, Djibouti from 2021, 02, 01 to 2021, 02, 28",
    "Show the pressure in Addis Ababa, Ethiopia from 2020, 05, 01 to 2020, 05, 31",
    "Display the sunshine duration in Nairobi, Kenya from 2021, 03, 01 to 2021, 03, 31",
    "Show the temperature in Lusaka, Zambia from 2020, 07, 01 to 2020, 07, 31",
    "Display the precipitation in Gaborone, Botswana from 2021, 09, 01 to 2021, 09, 30",
    "Highlight the snowfall in Harare, Zimbabwe from 2020, 01, 01 to 2020, 01, 31",
    "Show the wind speed in Windhoek, Namibia from 2021, 06, 01 to 2021, 06, 30",
    "Display the wind gust in Maputo, Mozambique from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the wind direction in Antananarivo, Madagascar from 2021, 12, 01 to 2021, 12, 31",
    "Show the pressure in Port Louis, Mauritius from 2020, 04, 01 to 2020, 04, 30",
    "Display the sunshine duration in Moroni, Comoros from 2021, 08, 01 to 2021, 08, 31",
    "Show the temperature in Mogadishu, Somalia from 2022, 01, 01 to 2022, 01, 31",
    "Display the precipitation in Djibouti, Djibouti from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the snowfall in Addis Ababa, Ethiopia from 2021, 12, 01 to 2021, 12, 31",
    "Show the wind speed in Nairobi, Kenya from 2020, 10, 01 to 2020, 10, 31",
    "Display the wind gust in Lusaka, Zambia from 2021, 08, 01 to 2021, 08, 31",
    "Highlight the wind direction in Gaborone, Botswana from 2021, 02, 01 to 2021, 02, 28",
    "Show the pressure in Harare, Zimbabwe from 2020, 05, 01 to 2020, 05, 31",
    "Display the sunshine duration in Windhoek, Namibia from 2021, 03, 01 to 2021, 03, 31",
    "Show the temperature in Maputo, Mozambique from 2020, 07, 01 to 2020, 07, 31",
    "Display the precipitation in Antananarivo, Madagascar from 2021, 09, 01 to 2021, 09, 30",
    "Highlight the snowfall in Port Louis, Mauritius from 2020, 01, 01 to 2020, 01, 31",
    "Show the wind speed in Moroni, Comoros from 2021, 06, 01 to 2021, 06, 30",
    "Display the wind gust in Mogadishu, Somalia from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the wind direction in Djibouti, Djibouti from 2021, 12, 01 to 2021, 12, 31",
    "Show the pressure in Addis Ababa, Ethiopia from 2020, 04, 01 to 2020, 04, 30",
    "Display the sunshine duration in Nairobi, Kenya from 2021, 08, 01 to 2021, 08, 31",
    "Show the temperature in Lusaka, Zambia from 2022, 01, 01 to 2022, 01, 31",
    "Display the precipitation in Gaborone, Botswana from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the snowfall in Harare, Zimbabwe from 2021, 12, 01 to 2021, 12, 31",
    "Show the wind speed in Windhoek, Namibia from 2020, 10, 01 to 2020, 10, 31",
    "Display the wind gust in Maputo, Mozambique from 2021, 08, 01 to 2021, 08, 31",
    "Highlight the wind direction in Antananarivo, Madagascar from 2021, 02, 01 to 2021, 02, 28",
    "Show the pressure in Port Louis, Mauritius from 2020, 05, 01 to 2020, 05, 31",
    "Display the sunshine duration in Moroni, Comoros from 2021, 03, 01 to 2021, 03, 31",
    "Show the temperature in Mogadishu, Somalia from 2020, 07, 01 to 2020, 07, 31",
    "Display the precipitation in Djibouti, Djibouti from 2021, 09, 01 to 2021, 09, 30",
    "Highlight the snowfall in Addis Ababa, Ethiopia from 2020, 01, 01 to 2020, 01, 31",
    "Show the wind speed in Nairobi, Kenya from 2021, 06, 01 to 2021, 06, 30",
    "Display the wind gust in Lusaka, Zambia from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the wind direction in Gaborone, Botswana from 2021, 12, 01 to 2021, 12, 31",
    "Show the pressure in Harare, Zimbabwe from 2020, 04, 01 to 2020, 04, 30",
    "Display the sunshine duration in Windhoek, Namibia from 2021, 08, 01 to 2021, 08, 31",
    "Show the temperature in Maputo, Mozambique from 2022, 01, 01 to 2022, 01, 31",
    "Display the precipitation in Antananarivo, Madagascar from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the snowfall in Port Louis, Mauritius from 2021, 12, 01 to 2021, 12, 31",
    "Show the wind speed in Moroni, Comoros from 2020, 10, 01 to 2020, 10, 31",
    "Display the wind gust in Mogadishu, Somalia from 2021, 08, 01 to 2021, 08, 31",
    "Highlight the wind direction in Djibouti, Djibouti from 2021, 02, 01 to 2021, 02, 28",
    "Show the pressure in Addis Ababa, Ethiopia from 2020, 05, 01 to 2020, 05, 31",
    "Display the sunshine duration in Nairobi, Kenya from 2021, 03, 01 to 2021, 03, 31",
    "Show the temperature in Lusaka, Zambia from 2020, 07, 01 to 2020, 07, 31",
    "Display the precipitation in Gaborone, Botswana from 2021, 09, 01 to 2021, 09, 30",
    "Highlight the snowfall in Harare, Zimbabwe from 2020, 01, 01 to 2020, 01, 31",
    "Show the wind speed in Windhoek, Namibia from 2021, 06, 01 to 2021, 06, 30",
    "Display the wind gust in Maputo, Mozambique from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the wind direction in Antananarivo, Madagascar from 2021, 12, 01 to 2021, 12, 31",
    "Show the pressure in Port Louis, Mauritius from 2020, 04, 01 to 2020, 04, 30",
    "Display the sunshine duration in Moroni, Comoros from 2021, 08, 01 to 2021, 08, 31",
    "Show the temperature in Mogadishu, Somalia from 2022, 01, 01 to 2022, 01, 31",
    "Display the precipitation in Djibouti, Djibouti from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the snowfall in Addis Ababa, Ethiopia from 2021, 12, 01 to 2021, 12, 31",
    "Show the wind speed in Nairobi, Kenya from 2020, 10, 01 to 2020, 10, 31",
    "Display the wind gust in Lusaka, Zambia from 2021, 08, 01 to 2021, 08, 31",
    "Highlight the wind direction in Gaborone, Botswana from 2021, 02, 01 to 2021, 02, 28",
    "Show the pressure in Harare, Zimbabwe from 2020, 05, 01 to 2020, 05, 31",
    "Display the sunshine duration in Windhoek, Namibia from 2021, 03, 01 to 2021, 03, 31",
    "Show the temperature in Maputo, Mozambique from 2020, 07, 01 to 2020, 07, 31",
    "Display the precipitation in Antananarivo, Madagascar from 2021, 09, 01 to 2021, 09, 30",
    "Highlight the snowfall in Port Louis, Mauritius from 2020, 01, 01 to 2020, 01, 31",
    "Show the wind speed in Moroni, Comoros from 2021, 06, 01 to 2021, 06, 30",
    "Display the wind gust in Mogadishu, Somalia from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the wind direction in Djibouti, Djibouti from 2021, 12, 01 to 2021, 12, 31",
    "Show the pressure in Addis Ababa, Ethiopia from 2020, 04, 01 to 2020, 04, 30",
    "Display the sunshine duration in Nairobi, Kenya from 2021, 08, 01 to 2021, 08, 31",
    "Show the temperature in Lusaka, Zambia from 2022, 01, 01 to 2022, 01, 31",
    "Display the precipitation in Gaborone, Botswana from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the snowfall in Harare, Zimbabwe from 2021, 12, 01 to 2021, 12, 31",
    "Show the wind speed in Windhoek, Namibia from 2020, 10, 01 to 2020, 10, 31",
    "Display the wind gust in Maputo, Mozambique from 2021, 08, 01 to 2021, 08, 31",
    "Highlight the wind direction in Antananarivo, Madagascar from 2021, 02, 01 to 2021, 02, 28",
    "Show the pressure in Port Louis, Mauritius from 2020, 05, 01 to 2020, 05, 31",
    "Display the sunshine duration in Moroni, Comoros from 2021, 03, 01 to 2021, 03, 31",
    "Show the temperature in Mogadishu, Somalia from 2020, 07, 01 to 2020, 07, 31",
    "Display the precipitation in Djibouti, Djibouti from 2021, 09, 01 to 2021, 09, 30",
    "Highlight the snowfall in Addis Ababa, Ethiopia from 2020, 01, 01 to 2020, 01, 31",
    "Show the wind speed in Nairobi, Kenya from 2021, 06, 01 to 2021, 06, 30",
    "Display the wind gust in Lusaka, Zambia from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the wind direction in Gaborone, Botswana from 2021, 12, 01 to 2021, 12, 31",
    "Show the pressure in Harare, Zimbabwe from 2020, 04, 01 to 2020, 04, 30",
    "Display the sunshine duration in Windhoek, Namibia from 2021, 08, 01 to 2021, 08, 31",
    "Show the temperature in Maputo, Mozambique from 2022, 01, 01 to 2022, 01, 31",
    "Display the precipitation in Antananarivo, Madagascar from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the snowfall in Port Louis, Mauritius from 2021, 12, 01 to 2021, 12, 31",
    "Show the wind speed in Moroni, Comoros from 2020, 10, 01 to 2020, 10, 31",
    "Display the wind gust in Mogadishu, Somalia from 2021, 08, 01 to 2021, 08, 31",
    "Highlight the wind direction in Djibouti, Djibouti from 2021, 02, 01 to 2021, 02, 28",
    "Show the pressure in Addis Ababa, Ethiopia from 2020, 05, 01 to 2020, 05, 31",
    "Display the sunshine duration in Nairobi, Kenya from 2021, 03, 01 to 2021, 03, 31",
    "Show the temperature in Lusaka, Zambia from 2020, 07, 01 to 2020, 07, 31",
    "Display the precipitation in Gaborone, Botswana from 2021, 09, 01 to 2021, 09, 30",
    "Highlight the snowfall in Harare, Zimbabwe from 2020, 01, 01 to 2020, 01, 31",
    "Show the wind speed in Windhoek, Namibia from 2021, 06, 01 to 2021, 06, 30",
    "Display the wind gust in Maputo, Mozambique from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the wind direction in Antananarivo, Madagascar from 2021, 12, 01 to 2021, 12, 31",
    "Show the pressure in Port Louis, Mauritius from 2020, 04, 01 to 2020, 04, 30",
    "Display the sunshine duration in Moroni, Comoros from 2021, 08, 01 to 2021, 08, 31",
    "Show the temperature in Mogadishu, Somalia from 2022, 01, 01 to 2022, 01, 31",
    "Display the precipitation in Djibouti, Djibouti from 2020, 11, 01 to 2020, 11, 30",
    "Highlight the snowfall in Addis Ababa, Ethiopia from 2021, 12, 01 to 2021, 12, 31",
    "Show the wind speed in Nairobi, Kenya from 2020, 10, 01 to 2020, 10, 31",
    "Display the wind gust in Lusaka, Zambia from 2021, 08, 01 to 2021, 08, 31",
    "Highlight the wind direction in Gaborone, Botswana from 2021, 02, 01 to 2021, 02, 28",
    "Show the pressure in Harare, Zimbabwe from 2020, 05, 01 to 2020, 05, 31",
    "Display the sunshine duration in Windhoek, Namibia from 2021, 03, 01 to 2021, 03, 31",
    "Show the temperature in Maputo, Mozambique from 2020, 07, 01 to 2020, 07, 31"
]

# Non-weather prompts (example)
non_weather_prompts = [
        "How's the weather today?",
    "What is the capital of France?",
    "What's the population of New York?",
    "Give me a list of countries in Europe",
    "Tell me a joke",
    "Who won the game last night?",
    "What's the time now?",
    "Play some music",
    "Set an alarm for 7 AM",
    "What is the latest news?",
    "What is your favorite book?",
    "What is the meaning of life?",
    "How do I cook pasta?",
    "What's your favorite movie?",
    "Tell me a story",
    "What is 2 + 2?",
    "How do I get to the nearest supermarket?",
    "What is the weather forecast for tomorrow?",
    "How do I change a tire?",
    "What's the best way to learn Python?",
    "What's your favorite color?",
    "What's the best pizza topping?",
    "How do I make a sandwich?",
    "What's the best way to exercise?",
    "How do I meditate?",
    "What are the benefits of yoga?",
    "How do I improve my memory?",
    "What's the best book you've read?",
    "How do I learn to play guitar?",
    "What is the stock market doing today?",
    "What are the latest trends in fashion?",
    "How do I bake a cake?",
    "What's the best way to save money?",
    "How do I start a garden?",
    "What are the symptoms of a cold?",
    "How do I improve my sleep?",
    "What is the best diet?",
    "How do I quit smoking?",
    "What is the history of the internet?",
    "How do I start a blog?",
    "What is the best way to travel on a budget?",
    "How do I stay motivated?",
    "What are the best apps for productivity?",
    "How do I improve my public speaking skills?",
    "What is the best way to learn a new language?",
    "What are the best career options?",
    "How do I negotiate a salary?",
    "What are the best investment strategies?",
    "How do I build a personal brand?",
    "What are the best ways to network?",
    "How do I write a resume?",
    "What are the best interview tips?",
    "How do I stay organized?",
    "What are the best time management techniques?",
    "How do I set goals?",
    "What are the benefits of volunteering?",
    "How do I improve my writing skills?",
    "What is the best way to handle stress?",
    "How do I develop a growth mindset?",
    "What are the best ways to relax?",
    "How do I improve my relationships?",
    "What are the best communication skills?",
    "How do I practice mindfulness?",
    "What are the best leadership qualities?",
    "How do I build self-confidence?",
    "What are the benefits of positive thinking?",
    "How do I overcome procrastination?",
    "What are the best ways to stay healthy?",
    "How do I create a workout plan?",
    "What are the best foods for energy?",
    "How do I boost my immune system?",
    "What are the best ways to stay hydrated?",
    "How do I create a balanced diet?",
    "What are the best ways to lose weight?",
    "How do I maintain a healthy weight?",
    "What are the best exercises for beginners?",
    "How do I build muscle?",
    "What are the best cardio workouts?",
    "How do I improve my flexibility?",
    "What are the benefits of strength training?",
    "How do I reduce belly fat?",
    "What are the best ways to recover after a workout?",
    "How do I stay motivated to exercise?",
    "What are the best ways to improve endurance?",
    "How do I prevent injuries during exercise?",
    "What are the best warm-up exercises?",
    "How do I cool down after a workout?",
    "What are the best ways to track fitness progress?",
    "How do I create a fitness routine?",
    "What are the benefits of HIIT workouts?",
    "How do I incorporate rest days into my fitness plan?",
    "What are the best ways to stay active at home?",
    "How do I make exercise fun?",
    "What are the best fitness challenges?",
    "How do I set fitness goals?",
    "What are the best exercises for weight loss?",
    "How do I stay consistent with my workouts?",
    "What are the benefits of group fitness classes?",
    "How do I find a workout buddy?",
    "What are the best online fitness resources?"
]



data = []
for i in range(1500):
    if random.random() < 0.5:
        prompt = random.choice(weather_prompts)
        label = 1
    else:
        prompt = random.choice(non_weather_prompts)
        label = 0
    data.append([prompt, label])

df = pd.DataFrame(data, columns=['text', 'label'])
df.to_csv('weather_prompts.csv', index=False)
print("weather_prompts.csv file created with 1500 rows of data.")

# Load and preprocess dataset
class PromptDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Load the dataset
from sklearn.model_selection import train_test_split
df = pd.read_csv('weather_prompts.csv')
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Initialize tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Create DataLoader
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = PromptDataset(
        texts=df.text.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4)

BATCH_SIZE = 16
MAX_LEN = 128

train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(val_df, tokenizer, MAX_LEN, BATCH_SIZE)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Fine-tune the model
EPOCHS = 3
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
loss_fn = torch.nn.CrossEntropyLoss().to(device)

# Training loop
for epoch in range(EPOCHS):
    model = model.train()
    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Evaluate the model
    model = model.eval()
    val_loss = 0
    val_accuracy = 0
    for batch in val_data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        loss = outputs.loss
        val_loss += loss.item()
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        accuracy = torch.sum(preds == labels).item() / len(labels)
        val_accuracy += accuracy

    val_loss /= len(val_data_loader)
    val_accuracy /= len(val_data_loader)
    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# Save the model
model.save_pretrained("fine-tuned-weather-classifier")
tokenizer.save_pretrained("fine-tuned-weather-classifier")

# Load the fine-tuned model for later use
weather_classification_pipeline = pipeline("text-classification", model="fine-tuned-weather-classifier", tokenizer="fine-tuned-weather-classifier")

# Sample CSV data for map generation
csv_data = {
    "Latitude": [40.7128, 34.0522, 41.8781],
    "Longitude": [-74.0060, -118.2437, -87.6298],
    "City": ["New York", "Los Angeles", "Chicago"],
    "Country": ["USA", "USA", "USA"],
    "Population": [8419000, 3980000, 2716000],
    "Elevation": [10, 71, 181]
}
df = pd.DataFrame(csv_data)

def parse_prompt(prompt):
    parsed_data = {
        "features": [],
        "csv_files": [df],  # Use the sample data
        "longitude": None,
        "latitude": None,
        "city": None,
        "country": None,
        "metric": None,
        "start_date": None,
        "end_date": None
    }

    classification_result = weather_classification_pipeline(prompt)[0]
    if classification_result['label'] == 'LABEL_0':  # Assuming LABEL_0 is non-weather
        print("Don't forget that this is a weather app.")
        return parsed_data

    # Improved extraction of weather metric, city, and country
    metric_match = re.search(r"(temperature|precipitation|snowfall|wind speed|wind gust|wind direction|pressure|sunshine duration)", prompt, re.IGNORECASE)
    if metric_match:
        parsed_data["metric"] = metric_match.group(0)

    city_country_match = re.search(r"in\s([\w\s]+),\s([\w\s]+)\sfrom", prompt, re.IGNORECASE)
    if city_country_match:
        parsed_data["city"] = city_country_match.group(1).strip()
        parsed_data["country"] = city_country_match.group(2).strip()

    date_match = re.findall(r"from\s(\d{4},\s\d{2},\s\d{2})\sto\s(\d{4},\s\d{2},\s\d{2})", prompt)
    if date_match:
        parsed_data["start_date"] = datetime.strptime(date_match[0][0], "%Y, %m, %d")
        parsed_data["end_date"] = datetime.strptime(date_match[0][1], "%Y, %m, %d")

    if not parsed_data["city"] or not parsed_data["country"] or not parsed_data["start_date"] or not parsed_data["end_date"]:
        print("Please provide valid city, country, and dates in your prompt.")
        return parsed_data

    # Geocode city and country to get latitude and longitude
    geolocator = Nominatim(user_agent="city_boundary_fetcher", timeout=10)
    location = geolocator.geocode(f"{parsed_data['city']}, {parsed_data['country']}")
    if location:
        parsed_data["latitude"], parsed_data["longitude"] = location.latitude, location.longitude
    else:
        print(f"Location not found for {parsed_data['city']}, {parsed_data['country']}.")

    return parsed_data

def fetch_weather_data(lat, lon, start_date, end_date):
    location = Point(lat, lon)
    data = Daily(location, start_date, end_date)
    data = data.fetch()
    return data

def plot_weather_data(parsed_data_list):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

    for parsed_data in parsed_data_list:
        if not parsed_data["latitude"] or not parsed_data["longitude"]:
            print("Location not found. Cannot plot the map.")
            continue

        ax.set_extent([parsed_data["longitude"] - 10, parsed_data["longitude"] + 10, parsed_data["latitude"] - 10, parsed_data["latitude"] + 10], crs=ccrs.PlateCarree())

        # Fetch weather data
        weather_data = fetch_weather_data(parsed_data["latitude"], parsed_data["longitude"], parsed_data["start_date"], parsed_data["end_date"])
        if weather_data.empty:
            print("No weather data available for the specified location and date range.")
            continue

        # Map the metric name to the corresponding column in the weather data
        metric_map = {
            "temperature": "tavg",
            "precipitation": "prcp",
            "snowfall": "snow",
            "wind speed": "wspd",
            "wind gust": "wpgt",
            "wind direction": "wdir",
            "pressure": "pres",
            "sunshine duration": "tsun"
        }
        metric = metric_map.get(parsed_data["metric"].lower(), None)
        if not metric:
            print(f"Invalid weather metric: {parsed_data['metric']}")
            continue

        # Plot the data
        if metric in weather_data.columns and not weather_data[metric].isnull().all():
            sc = ax.scatter([parsed_data["longitude"]], [parsed_data["latitude"]], c=[weather_data[metric].mean()], cmap='viridis', s=100, edgecolor='k', transform=ccrs.PlateCarree(), label=parsed_data["metric"])
            cb = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.05)
            cb.set_label(parsed_data["metric"].capitalize())

            ax.text(parsed_data["longitude"], parsed_data["latitude"], f"{parsed_data['city']}, {parsed_data['country']}", fontsize=12, transform=ccrs.PlateCarree())
        else:
            print(f"Metric {parsed_data['metric']} not available in the weather data.")

    ax.gridlines(draw_labels=True)
    plt.show()

# Main loop for terminal input
def main():
    parsed_data_list = []
    while True:
        prompt = input("Enter your weather prompt (or 'done' to finish): ")
        if prompt.lower() == 'done':
            if parsed_data_list:
                plot_weather_data(parsed_data_list)
            break
        parsed_data = parse_prompt(prompt)
        if parsed_data["latitude"] and parsed_data["longitude"]:
            parsed_data_list.append(parsed_data)

if __name__ == "__main__":
    main()

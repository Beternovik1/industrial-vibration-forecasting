# Industrial Centrifuge Condition Monitoring Dataset (CFG-42)

## Dataset Title
Industrial Centrifuge Condition Monitoring Dataset from Sulzer Centrifuge Unit at Arab Potash Company (Jordan)

## Description
This dataset contains one year of multisensor condition monitoring data collected from an industrial Sulzer centrifuge operating at the Arab Potash Company (APC) in Jordan.
The data were acquired using an ABB 800xA Distributed Control System (DCS) with synchronized one-minute sampling intervals.

The dataset reflects real industrial operating conditions, including natural load variations, seasonal thermal effects, and maintenance cycles. It is intended to support research in condition-based maintenance, predictive maintenance, prognostics and health management (PHM), digital twin modeling, reliability engineering, and industrial artificial intelligence.

The dataset is associated with the research article:
“Industrial Centrifuge Condition Monitoring Dataset from Sulzer Centrifuge Unit at Arab Potash Company (Jordan): A One-Year Multisensor Time Series for Predictive Maintenance and Reliability Studies.”

## Data Collection Period
Start: September 2022  
End: August 2023  
Duration: 12 months

## Data Acquisition System
Control System: ABB 800xA Distributed Control System (DCS)  
Sampling Interval: 1 minute  
Time Zone: UTC +3  

## Sensor Configuration
Torque sensor – Gear output shaft – % of nominal load  
Accelerometer – Rotor housing – mm/s (RMS)  
Thermocouple – Solid-end bearing – °C  
Thermocouple – Liquid-end bearing – °C  

## Dataset Files
CFG_42_EDS_Data.xlsx – Raw dataset  
CFG_42_EDS_Data.csv – Open-access dataset  
README_metadata.txt – Metadata file  

## Variables
Timestamp – Observation time  
Torque – % of rated load  
Vibration – mm/s RMS  
Temp_Solid – °C  
Temp_Liquid – °C  

## Alarm Thresholds
Torque ≥ 85 %  
Vibration ≥ 12 mm/s RMS  
Temperature ≥ 85 °C  

## License
Creative Commons Attribution 4.0 International (CC BY 4.0)

## Author
Doaa Ahmad Alqaraleh

## Corresponding Author
Email: Doua.ahmad90@yahoo.com

## Ethics Statement
This dataset does not involve human participants or animal subjects.

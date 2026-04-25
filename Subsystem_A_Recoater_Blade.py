###Master function, calculates active risk of failure
###Helper functions: gives constants for abrasive wear, contamination, production volume
###"On a long enough time line, the survival rate for everyone drops to zero." - Fight Club

from pathlib import Path
import math

file_path = Path("SystemA.txt")

# Create file if it doesn't exist
file_path.touch(exist_ok=True)

# Read content (will be empty if new)
content = file_path.read_text()




def abrasive_wear(content):
    Contamination = 0.5
    Humidity = 0.5
    Maintenance = 0.5
    Temperature = 0.5
    
    parts = content.split(";")
    time = 1
    ProductionVolume = 0

    if parts[0] != "":
        time = parts[0]
        ProductionVolume = parts[1]
        
    fr = failure_rate(0.5, Contamination, Humidity, Maintenance, Temperature)

    fail_x_time = fr * time

    fail_pwr_prodVol = fail_x_time ** (1 + ProductionVolume)

    fail_pwr_prodVol = fail_pwr_prodVol * -1

    result = 1 - math.exp(fail_pwr_prodVol)
    print(result)

    
    

    
def failure_rate(load, contamination, humidity, maintenance, temperature):
    numerator = load * (1 + contamination) * (1 + humidity)
    denominator = maintenance * (1 + temperature)
    failure_rate = numerator/ denominator
    print("The failure rate: " , failure_rate)
    return failure_rate

abrasive_wear(content)

import pandas as pd
import matplotlib.pyplot as plt
from colormath.color_objects import sRGBColor, LabColor, CMYKColor
from colormath.color_conversions import convert_color
from colorspacious import cspace_convert
from matplotlib.colors import rgb_to_hsv
import numpy as np

# Load the CSV file with HEX values
data = pd.read_csv('input dataset.csv')

# Extract HEX values from the 'HEXCode' column
hex_values = data['HEXCode']

# Remove '#' symbol from HEX codes
hex_values = hex_values.str.strip('#')

# Convert HEX to RGB
rgb_values = []

for hex_value in hex_values:
    try:
        # Check if the HEX code has a valid length
        if len(hex_value) == 6:
            # Attempt to convert the HEX code to RGB
            rgb_value = tuple(int(hex_value[i:i + 2], 16) for i in (0, 2, 4))
            rgb_values.append(rgb_value)
        else:
            # Skip invalid HEX codes
            print(f"Skipping invalid HEX code: {hex_value}")
    except ValueError:
        # Handle other conversion errors
        print(f"Skipping invalid HEX code: {hex_value}")

# Convert RGB to CMYK using colormath
cmyk_values = [convert_color(sRGBColor(*rgb), CMYKColor) for rgb in rgb_values]
cmyk_values = [(cmyk.cmyk_c, cmyk.cmyk_m, cmyk.cmyk_y, cmyk.cmyk_k) for cmyk in cmyk_values]

# Convert CMYK to RGB and scale to the 0-1 range
rgb_values_normalized = [(c[0]/100, c[1]/100, c[2]/100) for c in cmyk_values]

# Convert RGB to HSL
hsl_values = [rgb_to_hsv(np.array(rgb) / 255.0) for rgb in rgb_values]

# Plot CMYK with HEX color assignment
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter([c[0] for c in cmyk_values], [c[1] for c in cmyk_values], [c[2] for c in cmyk_values],
           c='black', s=10, marker='o', edgecolors='k')  # edgecolors='k' adds black borders
ax.set_xlabel('C', fontsize=8)
ax.set_ylabel('M', fontsize=8)
ax.set_zlabel('Y', fontsize=8)
ax.set_title('CMYK Color Space')

# Set all points to black
scatter.set_facecolor('k')

ax.tick_params(axis='both', which='both', labelsize=6)

plt.show()

# Plot HSL
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter([h[0] for h in hsl_values], [h[1] for h in hsl_values], [h[2] for h in hsl_values],
           c='black', s=10, marker='o', edgecolors='k')  # edgecolors='k' adds black borders
ax.set_xlabel('H', fontsize=8)
ax.set_ylabel('S', fontsize=8)
ax.set_zlabel('L', fontsize=8)
ax.set_title('HSL Color Space')

# Convert RGB to CIELAB using the provided function
def rgb2lab(input_color):
    num = 0
    RGB = [0, 0, 0]

    for value in input_color:
        value = float(value) / 255

        if value > 0.04045:
            value = ((value + 0.055) / 1.055) ** 2.4
        else:
            value = value / 12.92

        RGB[num] = value * 100
        num = num + 1

    XYZ = [0, 0, 0, ]

    X = RGB[0] * 0.4124 + RGB[1] * 0.3576 + RGB[2] * 0.1805
    Y = RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722
    Z = RGB[0] * 0.0193 + RGB[1] * 0.1192 + RGB[2] * 0.9505
    XYZ[0] = round(X, 4)
    XYZ[1] = round(Y, 4)
    XYZ[2] = round(Z, 4)

    XYZ[0] = float(XYZ[0]) / 95.047  # ref_X =  95.047   Observer= 2°, Illuminant= D65
    XYZ[1] = float(XYZ[1]) / 100.0  # ref_Y = 100.000
    XYZ[2] = float(XYZ[2]) / 108.883  # ref_Z = 108.883

    num = 0
    for value in XYZ:
        if value > 0.008856:
            value = value ** (0.3333333333333333)
        else:
            value = (7.787 * value) + (16 / 116)

        XYZ[num] = value
        num = num + 1

    Lab = [0, 0, 0]

    L = (116 * XYZ[1]) - 16
    a = 500 * (XYZ[0] - XYZ[1])
    b = 200 * (XYZ[1] - XYZ[2])

    Lab[0] = round(L, 4)
    Lab[1] = round(a, 4)
    Lab[2] = round(b, 4)

    return Lab

lab_values = [rgb2lab(rgb) for rgb in rgb_values]
lab_values = [(lab[1], lab[2], lab[0]) for lab in lab_values]

# Plot CIELAB
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter([l[0] for l in lab_values], [l[1] for l in lab_values], [l[2] for l in lab_values],
                     c='black', s=10, marker='o', edgecolors='k')  # edgecolors='k' adds black borders

# Assign original HEX colors to each point in CIELAB plot
scatter.set_facecolor([sRGBColor.new_from_rgb_hex(hex_value).get_value_tuple() for hex_value in hex_values])

ax.set_xlabel('a*', fontsize=8)
ax.set_ylabel('b*', fontsize=8)
ax.set_zlabel('L*', fontsize=8)
ax.set_title('CIELAB Color Space')

ax.tick_params(axis='both', which='both', labelsize=6)

plt.show()


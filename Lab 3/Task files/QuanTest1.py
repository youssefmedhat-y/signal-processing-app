def QuantizationTest1(file_name,Your_EncodedValues,Your_QuantizedValues):
    expectedEncodedValues=[]
    expectedQuantizedValues=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V2=str(L[0])
                V3=float(L[1])
                expectedEncodedValues.append(V2)
                expectedQuantizedValues.append(V3)
                line = f.readline()
            else:
                break
    if( (len(Your_EncodedValues)!=len(expectedEncodedValues)) or (len(Your_QuantizedValues)!=len(expectedQuantizedValues))):
        print("QuantizationTest1 Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_EncodedValues)):
        if(Your_EncodedValues[i]!=expectedEncodedValues[i]):
            print("QuantizationTest1 Test case failed, your EncodedValues have different EncodedValues from the expected one") 
            return
    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
            print("QuantizationTest1 Test case failed, your QuantizedValues have different values from the expected one") 
            return
    print("QuantizationTest1 Test case passed successfully")


def extract_values_from_file(file_path):
    """Extract encoded values and quantized values from a file."""
    encoded_values = []
    quantized_values = []
    
    with open(file_path, 'r') as f:
        # Skip the first 3 lines (header)
        for _ in range(3):
            f.readline()
        
        # Process the data lines
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(' ')
            if len(parts) == 2:
                encoded = parts[0]
                quantized = float(parts[1])
                
                encoded_values.append(encoded)
                quantized_values.append(quantized)
    
    return encoded_values, quantized_values


# Test your output against the expected output
if __name__ == "__main__":
    import os
    
    # Paths to the expected output and your output
    expected_output_path = "D:\\code\\python\\f1\\Lab 3\\Task files\\Quan1_Out.txt"
    your_output_path = "D:\\code\\python\\f1\\Lab 3\\myouts\\quan1out.txt"
    
    print(f"Loading values from your output file: {your_output_path}")
    your_encoded_values, your_quantized_values = extract_values_from_file(your_output_path)
    
    print(f"Your encoded values: {your_encoded_values}")
    print(f"Your quantized values: {your_quantized_values}")
    
    print("\nRunning QuantizationTest1...")
    QuantizationTest1(expected_output_path, your_encoded_values, your_quantized_values)
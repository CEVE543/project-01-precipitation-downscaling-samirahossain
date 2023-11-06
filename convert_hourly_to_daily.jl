using NCDatasets

# Define the full path to the input file
input_file = "data/raw/2m_temperature_1990.nc"

# Define the name of the output file for daily data
output_file = "data/raw/2m_temp_daily_1990.nc"

# Define the CDO command as a string with full paths
cdo_command = "cdo daymean $input_file $output_file"

# Run the CDO command using the run function
run(`$cdo_command`)
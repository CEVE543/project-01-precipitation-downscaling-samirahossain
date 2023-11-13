#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| output: false
using Dates
using MultivariateStats
using Plots
using NCDatasets
using StatsBase
using DataFrames
using Interpolations
#
#
#
#
#
#
#
#
observedprecip_ds= NCDataset("../data/precip_data_Houston/precip_tx.nc") #Storing predictand in observedprecip_ds

display(observedprecip_ds) #seeing how many attributes it has, check how many dimensions it has
obprecip_time=observedprecip_ds["time"][:];#finding about the dimension time in precip
obprecip_lon=observedprecip_ds["lon"][:];#stores longtiude of precip data
obprecip_lat=observedprecip_ds["lat"][:];#stores latitude of precip data
obprecip=observedprecip_ds["precip"][:,:,:];#stores precipitation values
display(obprecip_lat)
display(obprecip_lon)

obprecip_lat=reverse(obprecip_lat)#latitude data is reversed,thus we fix it 
obprecip=reverse(obprecip;dims=2)

using Plots 
heatmap(obprecip_lon,obprecip_lat,obprecip[:,:,5000];xlabel="Longitude",ylabel="Latitude",title="Precip on $(obprecip_time[5000]))")#visualize precipitation data of a day
#
#
#
#
#
start_date=Date(1990,1,1);
end_date=Date(2020,12,31);
time=Dates.Date.(obprecip_time);
display(time);
# display()
time_indices = findall(start_date .<= time .<= end_date);
display(time_indices);
obprecip_subset=obprecip[:,:,time_indices];
display(obprecip_subset)
heatmap(obprecip_lon,obprecip_lat,obprecip_subset[:,:,1];xlabel="Longitude",ylabel="Latitude",title="Precip on first day of 1990")
#
#
#
#
#
function open_mfdataset(files::Vector{String}, variable_name::AbstractString)

    # Lists to store variable data, time data, and other coordinate data
    var_data_list = []
    time_data_list = []
    coords_data_dict = Dict()

    # Open the first file to get the coordinate names (excluding time and the main variable)
    ds = Dataset(files[1])
    dimnames = keys(ds.dim)
    coord_names = setdiff(collect(dimnames), [variable_name, "time"])
    close(ds)

    # Initialize lists for each coordinate in coords_data_dict
    for coord in coord_names
        coords_data_dict[coord] = []
    end

    # Open each file, extract data, and store in lists
    for file in files
        ds = Dataset(file)

        # Store variable and time data
        push!(var_data_list, ds[variable_name][:])
        push!(time_data_list, ds["time"][:])

        # Store other coordinate data
        for coord in coord_names
            push!(coords_data_dict[coord], ds[coord][:])
        end

        close(ds)
    end

    # Pair variable data with time data and sort by time
    sorted_pairs = sort(collect(zip(time_data_list, var_data_list)); by=x -> x[1])
    sorted_time_data = [pair[1] for pair in sorted_pairs]
    sorted_var_data = [pair[2] for pair in sorted_pairs]

    # Concatenate sorted data
    concatenated_data_dict = Dict(
        variable_name => vcat(sorted_var_data...), "time" => vcat(sorted_time_data...)
    )

    # Concatenate coordinate data and add to the dictionary
    for coord in coord_names
        concatenated_data_dict[coord] = vcat(coords_data_dict[coord]...)
    end

    return concatenated_data_dict
end
#
#
#
#

using CDSAPI
using NCDatasets
using StatsBase: shuffle
base_path="data/raw" #folder where data were downloaded
files = ["2m_temperature_1990.nc", "2m_temperature_1991.nc", "2m_temperature_1992.nc","2m_temperature_1993.nc","2m_temperature_1994.nc","2m_temperature_1995.nc","2m_temperature_1996.nc","2m_temperature_1997.nc","2m_temperature_1998.nc","2m_temperature_1999.nc","2m_temperature_2000.nc","2m_temperature_2001.nc","2m_temperature_2002.nc","2m_temperature_2003.nc","2m_temperature_2004.nc","2m_temperature_2005.nc","2m_temperature_2006.nc","2m_temperature_2007.nc","2m_temperature_2008.nc","2m_temperature_2009.nc","2m_temperature_2010.nc","2m_temperature_2011.nc","2m_temperature_2012.nc","2m_temperature_2013.nc","2m_temperature_2014.nc","2m_temperature_2015.nc","2m_temperature_2016.nc","2m_temperature_2017.nc","2m_temperature_2018.nc","2m_temperature_2019.nc","2m_temperature_2020.nc"]
file_paths = [joinpath(base_path, file) for file in files];
data_dict = open_mfdataset(file_paths, "t2m")# data_dict stores the data in dictionary format
#
#
#
#
#
display(data_dict["longitude"]) #checking longitude of aggreated dataset
#
#
#
#
#
start_date = Date(data_dict["time"][1]);
end_date = Date(data_dict["time"][end]);

println(start_date);
println(end_date);

# Calculate the number of days between the two dates
num_days = (end_date) - (start_date) ;
num_days = Dates.value(num_days) + 1;
num_years = (Dates.year(end_date) - Dates.year(start_date)) + 1;

println("Number of days from $(Dates.year(start_date)) to $(Dates.year(end_date)): $num_days and $num_years years" );

n_lon = Int(length(data_dict["longitude"])/(num_years));#finding total number of longitude 
n_lat = Int(length(data_dict["latitude"])/(num_years));#finding total number of latitude

temp_long = data_dict["longitude"][1:n_lon];
temp_lat = data_dict["latitude"][1:n_lat];

println("n_lon = $n_lon ");
println("n_lat = $n_lat ");

data_dict_temp = reshape(data_dict["t2m"], (n_lon,n_lat,num_days*24));#data_dict is a long array, we convert it to 3 dimensional dataset
data_dict_time = data_dict["time"];

time_entries=[];
time_entries_all=[];
daily_temp = Array{Float64}(undef,n_lon,n_lat,1);#initializing an array to store the daily temperature data

#we use a for loop to extract the time entries from dictionary type data for temperature
for i in 1:length(data_dict_time)
    a = data_dict_time[i]
    year = Dates.year(a)
    month = Dates.month(a)
    day = Dates.day(a)
    time_entry = Date(year,month,day)
    push!(time_entries,time_entry)
end

display(time_entries);
calender = Date.(time_entries[1]:Day(1):time_entries[end]);
append!(time_entries_all,calender);
#
#
#

for date in calender
    indexArray = findall(x->x==date, time_entries); #finding the years 
    hourly_temp = data_dict_temp[:,:,indexArray]; #find the hourly temp
    average = mean(hourly_temp,dims=3) #take a mean of all hourly data for each year
    display(date);
    daily_temp = cat(daily_temp,average,dims =3) #we assign the average or daily temperature to be stored in a variable  
end

display(daily_temp);#the reshaped and daily air temperature data
#
#
#
daily_temp_data = copy(daily_temp[:,:,2:end]);#for some strange reason, there is extra year's data generated, which we omit here in this step
display(daily_temp_data)
temp_lat=reverse(temp_lat);#we reverse the latitude data to make the order ascending
daily_temp_reverse=reverse(daily_temp_data;dims=2);
display(daily_temp_reverse)
#
#
#
#

display(temp_long);
display(temp_lat);
heatmap(temp_long,temp_lat,transpose(daily_temp_reverse[:,:,20]),xlabel="Longitude",ylabel="Latitude",title="Temperature on $(time_entries_all[20])") #plotting heatmap for the 20th daily temperature across the region.
#
#
#
#
#
#In this code block, we are subsetting the daily air temperature data for Texas only.

display(temp_long);
temp_long_1=copy(temp_long);#this is unnecessary but might be helpful if we want to retain the original vector for longitude for comparison or change

lon_min = -108;#Texas's minimum longtiude, this is roughly taken from google map
lon_max = -93;#Texas's maximum longitude
i, j = findall(x -> x in [lon_min, lon_max], temp_long_1);
display(i);
display(j);

# Subset the array to include longitudes within the specified range
temp_long_tx = temp_long_1[(temp_long_1 .>= lon_min) .& (temp_long_1 .<= lon_max)];

temp_lat_1=copy(temp_lat);

lat_min = 26;#minimum latitude of Texas, from google map
lat_max = 37;#maximum latitude of Texas, from google map
a,b = findall(x -> x in [lat_min, lat_max], temp_lat_1);
display(a);
display(b);
# Subset the array to include longitudes within the specified range
temp_lat_tx = temp_lat_1[(temp_lat_1 .>= lat_min) .& (temp_lat_1 .<= lat_max)];
display(temp_long_tx)
display(temp_lat_tx)
#
#
#
#
#
daily_temp_tx=[];
daily_temp_1= copy(daily_temp_reverse);

display(daily_temp_1);
display(time_entries);
daily_temp_tx = daily_temp_1[i:j,a:b,:];
day = 200;
date = time_entries[day];
heatmap(temp_long_tx,temp_lat_tx,transpose(daily_temp_tx[:,:,day]),xlabel="Longitude",ylabel="Latitude",title="Temperature on $(date)")
#
#
#
#
#
#
#This is just a code block to see how weather changed over time for the whole North America. It is not necessary for this code, it's just a time lapse plot I found really cool to be able to code!

# daily_temp_tx=[]
# daily_temp_1= copy(daily_temp_reverse)
# display(daily_temp_1)
# display(time_entries)
# # daily_temp_tx = daily_temp_1[i:j,a:b,:]
# daily_temp_tx = daily_temp_1[:,:,:]


# for day in  1:1000
#     print(day,"\n")
#     date = time_entries_all[day]
#     h = heatmap(temp_long,temp_lat,transpose(daily_temp_tx[:,:,day]),xlabel="Longitude",ylabel="Latitude",title="Temperature on $(date)",clims=(250.,315.))
#     display(h)
#     sleep(0.01)
# end

#
#
#
#
#
#reading in dew point temperature data from ERA5 Reanalysis
using CDSAPI
using NCDatasets
using StatsBase: shuffle
base_path="data/raw1"
files1 = ["2m_dewpoint_temperature_1990.nc", "2m_dewpoint_temperature_1991.nc", "2m_dewpoint_temperature_1992.nc","2m_dewpoint_temperature_1993.nc","2m_dewpoint_temperature_1994.nc","2m_dewpoint_temperature_1995.nc","2m_dewpoint_temperature_1996.nc","2m_dewpoint_temperature_1997.nc","2m_dewpoint_temperature_1998.nc","2m_dewpoint_temperature_1999.nc","2m_dewpoint_temperature_2000.nc","2m_dewpoint_temperature_2001.nc","2m_dewpoint_temperature_2002.nc","2m_dewpoint_temperature_2003.nc","2m_dewpoint_temperature_2004.nc","2m_dewpoint_temperature_2005.nc","2m_dewpoint_temperature_2006.nc","2m_dewpoint_temperature_2007.nc","2m_dewpoint_temperature_2008.nc","2m_dewpoint_temperature_2009.nc","2m_dewpoint_temperature_2010.nc","2m_dewpoint_temperature_2011.nc","2m_dewpoint_temperature_2012.nc","2m_dewpoint_temperature_2013.nc","2m_dewpoint_temperature_2014.nc","2m_dewpoint_temperature_2015.nc","2m_dewpoint_temperature_2016.nc","2m_dewpoint_temperature_2017.nc","2m_dewpoint_temperature_2018.nc","2m_dewpoint_temperature_2019.nc","2m_dewpoint_temperature_2020.nc"];
file_paths1 = [joinpath(base_path, file) for file in files1];
data_dict_1 = open_mfdataset(file_paths1, "d2m");
#Check for missing values in them
#Check spatial and temporal dimensions, if it matches with your y dimensions

#
#
#
#
#converting the hourly dewpoint data to daily dewpoint data

start_date = Date(data_dict_1["time"][1]);
end_date = Date(data_dict_1["time"][end]);

println(start_date);
println(end_date);

# Calculate the number of days between the two dates
num_days = (end_date) - (start_date);
num_days = Dates.value(num_days) + 1 ;
num_years = (Dates.year(end_date) - Dates.year(start_date)) + 1 ;

println("Number of days from $(Dates.year(start_date)) to $(Dates.year(end_date)): $num_days and $num_years years" );

n_lon = Int(length(data_dict_1["longitude"])/(num_years));
n_lat = Int(length(data_dict_1["latitude"])/(num_years));
println("n_lon = $n_lon ");
println("n_lat = $n_lat ");

data_dict_dewtemp = reshape(data_dict_1["d2m"], (n_lon,n_lat,num_days*24));
data_dict_time1 = data_dict_1["time"];
time_entries=[];
time_entries_all=[];
daily_dewtemp = Array{Float64}(undef,n_lon,n_lat,1);

for i in 1:length(data_dict_time1)
    a = data_dict_time1[i]
    year = Dates.year(a)
    month = Dates.month(a)
    day = Dates.day(a)
    time_entry = Date(year,month,day)
    push!(time_entries,time_entry)
end
display(time_entries);
calender = Date.(time_entries[1]:Day(1):time_entries[end]);
append!(time_entries_all,calender);
for date in calender
    indexArray = findall(x->x==date, time_entries)
    # display(indexArray) 
    hourly_dewtemp = data_dict_dewtemp[:,:,indexArray]
    average1 = mean(hourly_dewtemp,dims=3)
    display(date)
    daily_dewtemp = cat(daily_dewtemp,average1,dims =3)
    # daily_dewtemp[:,:,j] = average[:,:,:]
    # j+=1
end

display(daily_dewtemp);
#
#
#
daily_dewtemp_data = copy(daily_dewtemp[:,:,2:end]);
display(daily_dewtemp_data);
#temp_lat=reverse(temp_lat)
daily_dewtemp_reverse=reverse(daily_dewtemp_data;dims=2);
#
#
#

display(temp_long);
display(temp_lat);
display(daily_dewtemp_data[:,:,:])#daily dew point data over North America

heatmap(temp_long,temp_lat,transpose(daily_dewtemp_reverse[:,:,20]),xlabel="Longitude",ylabel="Latitude",title="Dewpoint Temperature on $(time_entries_all[20])")
#
#
#
#
#subsetting the dew point daily data over Texas only
#These steps could have been squeezed into a function and would have resulted in shorter lines of code.
display(temp_long);
temp_long_1=copy(temp_long);

lon_min = -108;
lon_max = -93;
i, j = findall(x -> x in [lon_min, lon_max], temp_long_1);
display(i);
display(j);
# Subset the array to include longitudes within the specified range
temp_long_tx = temp_long_1[(temp_long_1 .>= lon_min) .& (temp_long_1 .<= lon_max)];

#display(temp_long_htx)


#display(temp_lat)
temp_lat_1=copy(temp_lat);

lat_min = 26;
lat_max = 37;
a,b = findall(x -> x in [lat_min, lat_max], temp_lat_1);
display(a);
display(b);
# Subset the array to include longitudes within the specified range
temp_lat_tx = temp_lat_1[(temp_lat_1 .>= lat_min) .& (temp_lat_1 .<= lat_max)];

#
#
#
#
daily_dewtemp_tx=[];
daily_dewtemp_1= copy(daily_dewtemp_reverse);
display(daily_temp_1);
display(time_entries);
daily_dewtemp_tx = daily_dewtemp_1[i:j,a:b,:];
day = 200;
date = time_entries[day];
heatmap(temp_long_tx,temp_lat_tx,transpose(daily_dewtemp_tx[:,:,day]),xlabel="Longitude",ylabel="Latitude",title="Dew Point Temperature on $(date)")
display(time_entries_all)
#
#
#
#
#
#
#
#
#
#
#This code block splits the data to training and testing sets

unique_years=unique(year.(time_entries_all));#we find the unique years in the time_entries
num_years=length(unique_years); #find the number of years
split_year=unique_years[end-9]; #specify split point
split_index=findfirst(x->year(x)==split_year,time_entries_all); #find split index
precip_train=obprecip_subset[:,:,1:split_index]; #split precipitation data
precip_test= obprecip_subset[:,:,(split_index+1):end];
temp_train= daily_temp_tx[:,:,1:split_index];#split air temperature data
temp_test=daily_temp_tx[:,:,(split_index+1):end];
dewtemp_train=daily_dewtemp_tx[:,:,1:split_index];#split dewpoint temperature data
dewtemp_test=daily_dewtemp_tx[:,:,(split_index+1):end];
display(dewtemp_train)
display(dewtemp_test)
#
#
#
#
#
#
#This function code block preprocesses the datasets by normalizing them and then reshapes them to 2D data structure from 3D structure.
function preprocess(temp::Array{T,3},temp_ref::Array{T,3})::AbstractMatrix where {T}
n_lon,n_lat,n_t=size(temp) #find longitude and latitude sizes
display(n_lon);
mean_value=mean(temp_ref;dims=3); #find mean of the input data along time dimension
std_dev=std(temp_ref;dims=3)#find standard deviation
temp_anom=(temp.-mean_value)./std_dev #rescale
temp_anom=reshape(temp_anom,n_lon * n_lat,n_t) #reshape to 2D structure
return(temp_anom)
end
#
#
#
#
#
#we call the preprocess function for the datasets
temp_mat_train=preprocess(temp_train,temp_train);
temp_mat_test=preprocess(temp_test,temp_train);
dewtemp_mat_train=preprocess(dewtemp_train,dewtemp_train);
dewtemp_mat_test=preprocess(dewtemp_test,dewtemp_train);
display(dewtemp_mat_train)
#
#
#
#
# temp_train_transposed=
#X_train= hcat(temp_mat_train,dewtemp_mat_train)
#X_test= hcat(temp_mat_test, dewtemp_mat_test)
X_train = cat(temp_mat_train, dewtemp_mat_train, dims=1);#concateneting the two training datasets
#
#
#
#
#PCA analysis
pca_model_both=fit(PCA,X_train;maxoutdim=30,pratio=0.999); #we first keep 30 principal components
display(principalvars(pca_model_both))
p2=plot(principalvars(pca_model_both)/var(pca_model_both);xlabel="# of PCs",ylabel="fraction of variance explained",label=false,title="Variance Explained")
#
#
#
#This is a trial code block to see the dimensions
#println(size(temp_lat_tx))
#println(size(temp_long_tx))
#println(size(pc))

#
#
#
#
#this is just a trial code block to visualizing the pcs
# p = []
# for i in 1:5
#     pc = projection(pca_model_both)[:, i]
#     pc = reshape(pc, length(temp_lat_tx),length(temp_long_tx))'
#     pi = heatmap(
#         temp_long_tx,  # Adjusted to match size of pc
#         temp_lat_tx,   # Adjusted to match size of pc
#         pc,
#         xlabel="Longitude",
#         ylabel="Latitude",
#         title="PC $i",
#         aspect_ratio=:equal,
#         cmap=:PuOr
#     )
#     push!(p, pi)
# end
# plot(p...; layout=(3, 2), size=(1500, 600))




#
#
#
#

p = []
for i in 1:5
    pc = projection(pca_model_both)[:, i]

    # Split into temperature and dewpoint parts
    pc_temp = pc[1:192]  # First half for temperature
    pc_dew = pc[193:end]  # Second half for dewpoint

    # Reshape each part
    pc_temp_reshaped = reshape(pc_temp, length(temp_lat_tx), length(temp_long_tx))'
    pc_dew_reshaped = reshape(pc_dew, length(temp_lat_tx), length(temp_long_tx))'

    # Create heatmaps for each part
    pi_temp = heatmap(
        temp_long_tx,
        temp_lat_tx,
        pc_temp_reshaped,
        xlabel="Longitude",
        ylabel="Latitude",
        title="Temp Component of PC $i",
        aspect_ratio=:equal,
        cmap=:PuOr
    )
    pi_dew = heatmap(
        temp_long_tx,
        temp_lat_tx,
        pc_dew_reshaped,
        xlabel="Longitude",
        ylabel="Latitude",
        title="Dewpoint Component of PC $i",
        aspect_ratio=:equal,
        cmap=:PuOr
    )

    push!(p, pi_temp)
    push!(p, pi_dew)
end
plot(p...; layout=(5, 4), size=(1500, 1200))


#
#
#
#
#

pc_ts = predict(pca_model_both, X_train)
day_of_year = Dates.dayofyear.(time)
p = []
for i in 1:5
pi = scatter(
            day_of_year,
            pc_ts[i, :];
            xlabel="Day of Year",
            ylabel="PC $i",
            title="PC $i",
            label=false,
            alpha=0.3,
            color=:gray
            )
push!(p, pi)
end
plot(p...; layout=(1, 5), size=(1500, 1000))


#
#
#
#
#
#
#

# pc_ts = predict(pca_model_both, temp_mat_train)
# day_of_year = Dates.dayofyear.(time)
# p = []
# for i in 1:5
# pi = scatter(
# day_of_year,
# pc_ts[i, :];
# xlabel="Day of Year",
# ylabel="PC $i",
# title="PC $i",
# label=true,
# alpha=0.3,
# color=:gray
# )
# push!(p, pi)
# end
# plot(p...; layout=(1, 5), size=(1500, 600))


#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# Function to calculate Euclidean distance between two vectors.
# @param x First vector.
# @param y Second vector.
# @return Euclidean distance as a floating-point number.
function euclidian_distance(x::AbstractVector,y::AbstractVector)::AbstractFloat
return sqrt(sum((x .- y).^2)) # Element-wise subtraction, squaring, summing, and square root calculation to get the distance
end


# Function to find indices of the 'n' smallest elements in a vector.
# @param x Vector in which to find the smallest elements.
# @param n Number of smallest elements to find.
# @return Vector of indices of the 'n' smallest elements.
function nsmallest(x::AbstractVector,n::Int)::Vector{Int}
  idx=sortperm(x) # Sorting the vector and returning the first 'n' indices
  return idx[1:n]
end



# K-Nearest Neighbors (KNN) function.
# @param X Matrix of data points.
# @param Xi Target vector for which to find neighbors.
# @param K Number of nearest neighbors to find.
# @return Tuple of sampled indices and corresponding data points from X.
function knn(X::AbstractMatrix,Xi::AbstractVector,K::Int)::Tuple{Int,AbstractVector}
dist=[euclidian_distance(Xi,X[j,:]) for j in 1:size(X,1)]  # Calculate distances from target vector to all points in X
idx=nsmallest(dist,K) # Find indices of the K nearest neighbors
w= 1 ./dist[idx] # Calculate and normalize weights based on distances
w./=sum(w)
# Sampling from the K nearest neighbors based on calculated weights
idx_sample = sample(idx, Weights(w))
return (idx_sample,vec(X[idx_sample,:]))  # Return the sampled indices and corresponding data points from X
end
#
#
#
# let 
# X = collect([-1 0 1 2 3 4 5 6 7 8 9 10]')
# Xi=[5.4]
# K=3
# samples=[]

# for i in 1:1000
# idx,X_sample=knn(X,Xi,K)
# push!(samples,X_sample[1])
# end
# histogram(samples;bins=vec(X).+0.5,label="Samples",normalize=:pdf,xticks=vec(X))
# dist=[0.4,0.6,1.4]
# w=1 ./dist
# w./=sum(w)
# scatter!([5, 6, 4],w;label="Analytical",markersize=7)
# end

#
#
#
#
#
function predict_knn_combined(temp_train,temp_test,dewtemp_train,dewtemp_test,precip_train;n_pca::Int)

temp_mat_train=preprocess(temp_train,temp_train);
temp_mat_test=preprocess(temp_test,temp_train);
dewtemp_mat_train=preprocess(dewtemp_train,dewtemp_train);
dewtemp_mat_test=preprocess(dewtemp_test,dewtemp_train);

#X_train= hcat(temp_mat_train,dewtemp_mat_train)
#X_test= hcat(temp_mat_test, dewtemp_mat_test)

X_train = cat(temp_mat_train, dewtemp_mat_train, dims=1);
X_test = cat(temp_mat_test, dewtemp_mat_test, dims=1);

pca_model= fit(PCA,X_train;maxoutdim=n_pca);

train_embedded = predict(pca_model,X_train);
test_embedded = predict(pca_model,X_test);


println("Dimensions of train_projected: $(size(train_projected))");


precip_predict = map(1:size(X_test,2)) do i
idx,_=knn(train_embedded',test_embedded[:,i],2)

display(idx);
precip_train[:,:,idx];
end

return precip_predict
end

#
#
#
#this is a trial code block to see if the knn works 
# pca_model=fit(PCA,X_train;maxoutdim=5)
# train_projected=predict(pca_model,X_train)
# test_projected=predict(pca_model,X_test)

# precip_pred_combined=map(1:size(X_test,2)) do i
# idx,_=knn(train_projected',test_projected[:,1],3)
# display(idx)
# precip_train[:,:,idx]
# end
```
#
display(train_projected);
#
#
#
#
#
#
#
#
#
#
#
t_sample_combined1 = rand(1:size(temp_test, 3), 3);
precip_pred_combined1 = predict_knn_combined(temp_train, temp_test[:, :, t_sample_combined1], dewtemp_train, dewtemp_test[:, :, t_sample_combined1], precip_train; n_pca=10); #we call predict-knn-combined function to predict precipitation
display(precip_pred_combined1)

#
#
#
t_sample = rand(1:size(X_train, 3),3);
p = map(eachindex(t_sample)) do ti
t = t_sample[ti]
display(t_sample);
y_pred = precip_pred_combined[ti]'
y_actual = precip_test[:, :, t]'
cmax = max(maximum(skipmissing(y_pred)), maximum(skipmissing(y_actual)))

p1 = heatmap(
obprecip_lon,
obprecip_lat,
y_pred;
xlabel="Longitude",
ylabel="Latitude",
title="Predicted",
aspect_ratio=:equal,
clims=(0, cmax)
)
p2 = heatmap(
obprecip_lon,
obprecip_lat,
y_actual;
xlabel="Longitude",
ylabel="Latitude",
title="Actual",
aspect_ratio=:equal,
clims=(0, cmax)
)
plot(p1, p2; layout=(2, 1), size=(1000, 400))
end
plot(p...; layout=(2, 3), size=(1500, 1200))
#
#
#
#
# function predict_knn(temp_train, temp_test, precip_train; n_pca::Int)
# X_train1 = preprocess(temp_train, temp_train)
# X_test1 = preprocess(temp_test, temp_train)
# # fit the PCA model to the training data
# pca_model = fit(PCA, X_train; maxoutdim=n_pca)
# # project the test data onto the PCA basis
# train_embedded = predict(pca_model, X_train1)
# test_embedded = predict(pca_model, X_test1)
# display(train_embedded)
# # use the `knn` function for each point in the test data
# precip_pred = map(1:size(X_test1, 2)) do i
# idx, _ = knn(train_embedded', test_embedded[:, i], 3)
# display(idx)
# precip_train[:, :, idx]
# end
# return precip_pred
# end
#
#
#
#

# t_sample = rand(1:size(temp_test, 3), 3)
# precip_pred = predict_knn(temp_train, temp_test[:, :, t_sample], precip_train; n_pca=3)

# p = map(eachindex(t_sample)) do ti
# t = t_sample[ti]
# display(t_sample)
# y_pred = precip_pred[ti]'
# y_actual = precip_test[:, :, t]'
# cmax = max(maximum(skipmissing(y_pred)), maximum(skipmissing(y_actual)))

# p1 = heatmap(
# obprecip_lon,
# obprecip_lat,
# y_pred;
# xlabel="Longitude",
# ylabel="Latitude",
# title="Predicted",
# aspect_ratio=:equal,
# clims=(0, cmax)
# )
# p2 = heatmap(
# obprecip_lon,
# obprecip_lat,
# y_actual;
# xlabel="Longitude",
# ylabel="Latitude",
# title="Actual",
# aspect_ratio=:equal,
# clims=(0, cmax)
# )
# plot(p1, p2; layout=(2, 1), size=(1000, 400))
# end
# plot(p...; layout=(2, 3), size=(1500, 1200))


#
#
#
# t_sample = rand(1:size(dewtemp_test, 3), 3)
# precip_pred = predict_knn(dewtemp_train, dewtemp_test[:, :, t_sample], precip_train; n_pca=3)
#
#
#
#
#
using Statistics
mse = mean([mean(skipmissing((precip_pred_combined1[i] .- precip_test[:, :, i]).^2)) for i in 1:length(precip_pred_combined1)])
#
#
#
# Assuming precip_train is a 3D array (lon, lat, time)
n_time_points = size(precip_train, 3)
mean_precip = sum([mean(skipmissing(precip_train[:, :, t])) for t in 1:n_time_points if any(!ismissing, precip_train[:, :, t])]) / n_time_points


# Repeat the mean value across the spatial dimensions and time points
precip_pred_baseline = repeat(reshape([mean_precip], 1, 1), size(precip_test, 1), size(precip_test, 2), size(precip_test, 3))

# Calculating MSE, ensuring to handle missing values
mse_baseline = mean(skipmissing((precip_pred_baseline .- precip_test).^2))


#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

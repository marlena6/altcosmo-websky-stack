# modified from Zack Li

using Pixell, WCS, XGPaint
using Cosmology
using Interpolations
import XGPaint: AbstractProfile
using HDF5
import JSON
using JLD2
using ThreadsX
using FileIO
using Healpix
using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--halofile"
            help = "Halo file in hdf5 format"
            arg_type = String
            default = "/mnt/raid-cita/mlokken/buzzard/catalogs/halos/buzzard_halos.hdf5"
        "--OMstring"
            help = "Omega_m string"
            arg_type = String
            default = "OM32"
        "--Omega_c"
            help = "Omega_c"
            arg_type = Float64
            default = 0.24
        "--Omega_b"
            help = "Omega_b"
            arg_type = Float64
            default = 0.046
        "--h"
            help = "h"
            arg_type = Float64
            default = 0.7
    end
    return parse_args(s)
end

@show parsed_args = parse_commandline()

halofile = parsed_args["halofile"]
OMstring = parsed_args["OMstring"]
Omega_c = parsed_args["Omega_c"]
Omega_b = parsed_args["Omega_b"]
h = parsed_args["h"]

print("Threads: ", Threads.nthreads(), "\n")
modeltype::String = "battaglia"


shape, wcs = fullsky_geometry(0.5 * Pixell.arcminute)
# for a box:
# box = [30   -30;           # RA
#        -25     25] * Pixell.degree  # DEC
# shape, wcs = geometry(CarClenshawCurtis{Float64}, box, 0.5 * Pixell.arcminute)
m = Enmap(zeros(shape), wcs)
# precomputed sky angles
α_map, δ_map = posmap(shape, wcs)
psa = (sin_α=sin.(α_map), cos_α=cos.(α_map), sin_δ=sin.(δ_map), cos_δ=cos.(δ_map))

fid = h5open(halofile, "r")
ra, dec = deg2rad.(fid["ra"]), deg2rad.(fid["dec"])
halo_mass = collect(fid["m200c"])
redshift = collect(fid["z"])
# fill array with true
valid_halos = fill(true, length(ra))
# cut out halos that are within 20 degrees of the poles
# valid_halos_dec = (dec .> deg2rad(-70)) .& (dec .< deg2rad(70))
# valid_halos = valid_halos .& valid_halos_dec
# cut out halos within 10 degrees of 180 degrees
# valid_halos_ra = (ra .> deg2rad(-170)) .& (ra .< deg2rad(170))
# valid_halos = valid_halos .& valid_halos_ra
# and limit halo masses above 10^13, because Websky doesn't resolve below
in_halo_mass = (halo_mass .> 10^13)
valid_halos = valid_halos .& in_halo_mass
ra = ra[valid_halos]
dec = dec[valid_halos]
halo_mass =  halo_mass[valid_halos]
redshift = redshift[valid_halos]
close(fid)
print(rad2deg(minimum(ra)), " ", rad2deg(maximum(ra)), " ")
print(rad2deg(minimum(dec)), " ", rad2deg(maximum(dec)), " ")

# choose the cutoff for the pressure profiles here
cutoff = 4 # the standard
apply_beam = true
##
perm = sortperm(dec, alg=ThreadsX.MergeSort)
ra = ra[perm]
dec = dec[perm]
redshift = redshift[perm]
halo_mass = halo_mass[perm]



##
print("Precomputing the model profile grid.\n")

# set up a profile to paint
if modeltype=="break"
    p = XGPaint.BreakModel(Omega_c=Omega_c, Omega_b=Omega_b, h=h, alpha_break=1.5) # Buzzard cosmology
elseif modeltype=="battaglia"
    p = XGPaint.BattagliaProfile(Omega_c=Omega_c, Omega_b=Omega_b, h=h) # Buzzard cosmology
end

# beam stuff
N_logθ = 512
rft = RadialFourierTransform(n=N_logθ, pad=256)
beamstr = ""

if apply_beam
    ells = rft.l
    fwhm = deg2rad(1.6/60.)  # ACT beam
    σ² = fwhm^2 / (8log(2))
    lbeam = @. exp(-ells * (ells+1) * σ² / 2)
    beamstr = "_1p6arcmin"
end

model_file::String = "/mnt/raid-cita/mlokken/response_function_profiles/cached_$(modeltype)_$(beamstr)_$(OMstring).jld2"
if isfile(model_file)
    print("Found cached Battaglia profile model. Loading from disk.\n")
    model = load(model_file)
    prof_logθs, prof_redshift, prof_logMs, prof_y = model["prof_logθs"], 
        model["prof_redshift"], model["prof_logMs"], model["prof_y"]
else
    print("Didn't find a cached profile model. Computing and saving.\n")
    logθ_min, logθ_max = log(minimum(rft.r)), log(maximum(rft.r))
    @time prof_logθs, prof_redshift, prof_logMs, prof_y = profile_grid(p; 
        N_logθ=length(rft.r), logθ_min=logθ_min, logθ_max=logθ_max)
    
    if apply_beam
        # now apply the beam
        @time XGPaint.transform_profile_grid!(prof_y, rft, lbeam)
        prof_y = prof_y[begin+rft.pad:end-rft.pad, :, :]
        prof_logθs = prof_logθs[begin+rft.pad:end-rft.pad]
        XGPaint.cleanup_negatives!(prof_y)
    end
    jldsave(model_file; prof_logθs, prof_redshift, prof_logMs, prof_y)
    
end


itp = Interpolations.interpolate(log.(prof_y), BSpline(Cubic(Line(OnGrid()))))
sitp = scale(itp, prof_logθs, prof_redshift, prof_logMs);


##
function paint_map!(m::Enmap, p::XGPaint.AbstractProfile, psa, sitp, masses, 
                    redshifts, αs, δs, irange; mult=4)
    for i in irange
        α₀ = αs[i]
        δ₀ = δs[i]
        mh = masses[i]
        z = redshifts[i]
        θmax = XGPaint.θmax(p, mh * XGPaint.M_sun, z, mult=mult)
        profile_paint!(m, α₀, δ₀, psa, sitp, z, mh, θmax)
    end
end


function chunked_paint!(m::Enmap, p::XGPaint.AbstractProfile, psa, sitp, masses, 
                        redshifts, αs, δs; mult=4)
    m .= 0.0
    
    N_sources = length(masses)
    chunksize = ceil(Int, N_sources / (2Threads.nthreads()))
    chunks = chunk(N_sources, chunksize);
    
    Threads.@threads for i in 1:Threads.nthreads()
        chunk_i = 2i
        i1, i2 = chunks[chunk_i]
        paint_map!(m, p, psa, sitp, masses, redshifts, αs, δs, i1:i2, mult=mult)
    end

    Threads.@threads for i in 1:Threads.nthreads()
        chunk_i = 2i - 1
        i1, i2 = chunks[chunk_i]
        paint_map!(m, p, psa, sitp, masses, redshifts, αs, δs, i1:i2, mult=mult)
    end
end
print("Painting map.\n")

@time chunked_paint!(m, p, psa, sitp, halo_mass, redshift, ra, dec, mult=cutoff) # for an Enmap


write_map(
    "/mnt/raid-cita/mlokken/pkpatch/ymaps/$(split(halofile, "/")[end][1:end-5])_$(modeltype)_car$(beamstr)_cutoff$(cutoff).fits",
    m)

# using PyPlot
# plt.clf()
# plt.figure()
# plt.imshow(log10.(m.data'))
# plt.axis("off")
# plt.savefig("test_Healpix.png", bbox_inches="tight",pad_inches = 0)
# plt.gcf()



# ##
# m .= 0
# profile_paint!(m, 0., 0., psa, sitp, 0.1, 1e14, π/90)

# ##
# using Plots
# Plots.plot(log10.(m))

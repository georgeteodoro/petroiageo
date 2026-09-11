# General

The inverted learning system consists of a Python3 application that expands the hypercube of points, generating new porosity values. The application supports serialized execution (1 CPU core) as well as local and distributed parallelism via MPI.

# Versioning

To facilitate the process of resolving potential code issues, it is requested that only tags be used. Although the main branch is up to date most of the time, this is not guaranteed at all times, especially during delivery periods, when updates may be made.

# Requirements - Libraries

The following libraries are required to run the application in a Linux environment:

 - python3
 - python3-pip
 - python3-dev
 - mpi
 - libopenmpi-dev

The application has been tested with OpenMPI, but other MPI implementations may also be used.

## Requirements - Libraries - HDF5 and h5py

The hdf5 and h5py libraries are used for managing large files through Out-of-Core execution. To allow their use together with MPI, these two tools must be compiled manually. Root access is not required to install these libraries, except for the pip3 commands. However, they can be installed using a virtual environment, such as anaconda or venv, without root permissions.

These libraries have 2 scripts to assist with their installation: latest_install.sh and prep_env.sh. The second script assists in creating a venv when root access is unavailable. The operations are explained step by step below.

HDF5 can be compiled as follows:

	git clone https://github.com/HDFGroup/hdf5.git
	cd hdf5
	git checkout hdf5-1_12_2-3-rc1
	autoconf
	./configure --enable-parallel --enable-shared --prefix=<HDF5_PATH>/build
	make -j8
	make install


The compilation/installation of h5py, which should be performed after HDF5, can be done using the following commands:

	pip3 uninstall mpi4py
	pip3 install mpi4py==4.0.1
	git clone https://github.com/h5py/h5py.git
	cd h5py
	git checkout 3.11.0
	pip3 install wheel Cython==3.1.0a1
	export CC=mpicc; export HDF5_MPI="ON"; export HDF5_DIR="<HDF5_PATH>/build"; pip3 install --no-build-isolation .


h5py may have compatibility issues with mpi4py. To avoid problems, make sure that the mpi4py version is the one specified below. This is done in the commands above, where a previously installed version of mpi4py is removed and the expected version is installed.

### Compatibility List

The following versions have been successfully tested for compatibility:

 - python 3.12.7:
   - openmpi 4.1.6
   - mpi4py 4.0.1
   - HDF5 hdf5-1_12_2-3-rc1
   - Cython 3.1.0a1
   - h5py 3.11.0
 - python 3.12.3:
   - mpi4py 3.1.6
   - HDF5 hdf5-1_12_2-3-rc1
   - Cython 3.0.10
   - h5py 3.11.0
 - python 3.12.4:
   - mpi4py 3.1.6
   - HDF5 hdf5-1_12_2-3-rc1
   - Cython 3.0.10
   - h5py 3.11.0
   - openmpi 4.0.1 (sequana for sdumont)
   - gcc 13.2 (sequana for sdumont)

Note that using more recent version combinations is recommended, as there may have been issues with previous combinations, making it necessary to create a new combination.

# Requirements - Data

The application uses two types of data: seismic data and actual well porosity data. The actual well data is small, and therefore is already available in this private repository. The seismic data is larger and is therefore not included in the repository. These numpy files must be placed in the ./data/ directory, located at the root of this project. The following is the list of files required to run the application that must be placed in the ./data directory:
 - porosity-canal.npy
 - features/NEAR.npy
 - features/MID.npy
 - features/FAR.npy
 - features/UFAR.npy

The seismic data (NEAR.npy, MID.npy, FAR.npy, and UFAR.npy) consists of numpy files containing a matrix of (434,646,251) points (the hypercube dimensions), with values of type numpy.float64.

## Generating h5 Files

The .npy data must be converted to the .h5 format. This is done using the following commands, executed from the project root:

	cd src/data
	python3 wells_data3_hdf5.py
	cd features
	python3 seismic_data3_hdf5.py


The commands above generate .h5 files that should be located in the directories shown below:
 - ./data/POV/porosity-canal.h5
 - ./data/POV/features/h5_features/NEAR.npy
 - ./data/POV/features/h5_features/MID.npy
 - ./data/POV/features/h5_features/FAR.npy
 - ./data/POV/features/h5_features/UFAR.npy

The Python scripts already generate the data in the correct locations, but it is important to verify that the files exist.

# Python Dependencies

The application uses several Python libraries for its execution. All of them (except h5py, which was installed previously) can be easily installed using the following command:

	pip3 install -r requirements

It is worth noting that these dependencies should only be installed after the required Linux libraries.

If root access is unavailable, this installation can be performed in a virtual environment supported by the execution environment, such as anaconda or venv.

# Execution

There are two ways to run the application: serialized execution using a single process, or parallel execution. Given the computational cost involved, serialized execution is recommended only for validating the application installation. Serialized execution is performed using the following command from the ./src/alg directory of this repository:

	python3 main.py --config config.yaml [params]


For parallel execution, simply run the following command to generate N parallel processes:

	mpirun -np N --tag-output --bind-to core python3 main.py --config config.yaml [params]


These N parallel processes are distributed across the available resources. Therefore, on a single machine with 10 cores and N=20, 20 processes are initialized on that machine. If there are two machines configured in a distributed environment, there will be 10 processes per machine. It is not recommended to use more processes per machine than the number of CPU cores it has.

Another point of attention is that this application consumes a large amount of memory, and therefore very large values of N may result in executions being interrupted due to insufficient memory. *Note: this has been mostly solved with shared memory TrialDara*

The porosity estimation data is saved in ./data/POV/porosity-canal.h5. Data related to application performance (e.g., execution times) and the performance of the generated model (e.g., model accuracy) can be seen in the system's standard output (stdout).

# Execution Parameters

In this version of the application, there are input parameters that can be used to configure how data generation is performed. The application also has a "help" parameter: --help or -h. Some available parameters are listed below:

 - --it: Initial iteration to be executed. For example, if the value is 3, it is expected that iterations 1 and 2 have already been completed, with their results stored in the porosity file; therefore, iteration 3 onward will be executed. Note: the first iteration is iteration 1.
 - --nits: Defines the number of iterations to be performed, including the initial iteration. If the application is unable to finish its execution (e.g., it is canceled due to excessive memory usage or a timeout on clusters), the partial data from that execution is LOST. To avoid this problem in cluster environments, it is recommended to execute only a few iterations at a time and back up the porosity files after each completion.
 - --nf: Number of features to be used, based on the available base features in ./data/POV/features/h5_features. If not defined, all available base features are used.
 - --nfs: Number of features selected per iteration. By default, 10 features are used to generate a porosity estimation model. However, this value can be reduced to decrease execution time.

In addition, a configuration file (config.yaml) is used to configure the application at a finer level. If a parameter is passed via a Python argument that already has a value defined in config.yaml, the configuration value is overridden, and the value passed as an argument is retained.

# Example

Below is a quick example using only 1 file, selecting only 1 best feature, for 1 iteration:

	python3 main.py --config config.yaml --nits 1 --nf 1 --nsf 1


The output of the command above will be:

	loading seismic
	loading porosity
	[gen_expanded_points][it0] Expanding points on ring 0
	[PROFILING][expand][it0][chunk0-time] 5.728178262710571
	[PROFILING][expand][it0][chunk1-time] 6.6458210945129395
	[PROFILING][expand][it0][chunk0-time] 5.793959140777588
	[PROFILING][expand][it0][chunk1-time] 5.810708999633789
	[PROFILING][expand][it0][chunk0-time] 5.762479782104492
	[PROFILING][expand][it0][chunk0-time] 5.958775043487549
	[PROFILING][expand][it0][chunk0-time] 5.750436782836914
	[PROFILING][expand][it0][chunk0-time] 5.760072231292725
	[PROFILING][expand][it0][chunk0-time] 5.853853464126587
	[PROFILING][expand][it0][chunk0-time] 5.790080308914185
	[PROFILING][expand][it0][ran-chunks-time] 58.85436511039734
	[PROFILING][expand][it0][chunks-ran] 1 2
	============== NEED TO AUTOMATE TMP_LIST CHUNK_SIZE
	[get_features_sets][('MID', -3, -3, -3)] insert_feature_time: 0.0773627758026123
	[get_features_sets][('MID', -3, -3, -3)] train_time: 0.4123711585998535
	[get_features_sets][('MID', -3, -3, -3)] error: 0.05177651273487806
	[get_features_sets][it0] Tested features ['x', 'y', 'z', ('MID', -3, -3, -3)] with error 0.05177651273487806
	[get_features_sets][it0] full_it_time: 0.48976802825927734
	[get_features_sets][('MID', -3, -3, -3)] commit_feature_time: 0.0767519474029541
	[get_features_sets] full_time: 3.45944881439209

# Execution via Container

Another execution option is through the Dockerfile available in this repository. Using the container, it is possible to run the application without having to worry about downloading the required modules/dependencies and without requiring root access. Two commands are required (executed from the root of this repository):

	podman build --tag app .
	podman run -v <GIT_PATH>/data:/home/petroiageo/data:Z app


The first command generates the container image to be executed. In the second command, the absolute path to this repository (GIT_PATH) must be provided. Application parameters can normally be passed to the application via the CLI. The seismic and actual porosity data must be present in the ./data directory on the host system. Therefore, no data movement (copying) between the container image and the host system is required. This example was tested using podman for container creation and execution; however, since this is a Dockerfile, it can be executed using the container software of your choice.

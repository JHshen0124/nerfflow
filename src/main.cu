/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   main.cu
 *  @author Thomas Müller, NVIDIA
 */

// #include <neural-graphics-primitives/testbed.h>

// #include <tiny-cuda-nn/common.h>

// #include <args/args.hxx>

// #include <filesystem/path.h>

// using namespace args;
// using namespace ngp;
// using namespace std;
// using namespace tcnn;
// namespace fs = ::filesystem;

// int main(int argc, char** argv) {
// 	ArgumentParser parser{
// 		"neural graphics primitives\n"
// 		"version " NGP_VERSION,
// 		"",
// 	};

// 	HelpFlag help_flag{
// 		parser,
// 		"HELP",
// 		"Display this help menu.",
// 		{'h', "help"},
// 	};

// 	ValueFlag<string> mode_flag{
// 		parser,
// 		"MODE",
// 		"Mode can be 'nerf', 'sdf', or 'image' or 'volume'. Inferred from the scene if unspecified.",
// 		{'m', "mode"},
// 	};

// 	ValueFlag<string> network_config_flag{
// 		parser,
// 		"CONFIG",
// 		"Path to the network config. Uses the scene's default if unspecified.",
// 		{'n', 'c', "network", "config"},
// 	};

// 	Flag no_gui_flag{
// 		parser,
// 		"NO_GUI",
// 		"Disables the GUI and instead reports training progress on the command line.",
// 		{"no-gui"},
// 	};

// 	Flag no_train_flag{
// 		parser,
// 		"NO_TRAIN",
// 		"Disables training on startup.",
// 		{"no-train"},
// 	};

// 	ValueFlag<string> scene_flag{
// 		parser,
// 		"SCENE",
// 		"The scene to load. Can be NeRF dataset, a *.obj mesh for training a SDF, an image, or a *.nvdb volume.",
// 		{'s', "scene"},
// 	};

// 	ValueFlag<string> snapshot_flag{
// 		parser,
// 		"SNAPSHOT",
// 		"Optional snapshot to load upon startup.",
// 		{"snapshot"},
// 	};

// 	ValueFlag<uint32_t> width_flag{
// 		parser,
// 		"WIDTH",
// 		"Resolution width of the GUI.",
// 		{"width"},
// 	};

// 	ValueFlag<uint32_t> height_flag{
// 		parser,
// 		"HEIGHT",
// 		"Resolution height of the GUI.",
// 		{"height"},
// 	};

// 	Flag version_flag{
// 		parser,
// 		"VERSION",
// 		"Display the version of neural graphics primitives.",
// 		{'v', "version"},
// 	};

// 	// Parse command line arguments and react to parsing
// 	// errors using exceptions.
// 	try {
// 		parser.ParseCLI(argc, argv);
// 	} catch (const Help&) {
// 		cout << parser;
// 		return 0;
// 	} catch (const ParseError& e) {
// 		cerr << e.what() << endl;
// 		cerr << parser;
// 		return -1;
// 	} catch (const ValidationError& e) {
// 		cerr << e.what() << endl;
// 		cerr << parser;
// 		return -2;
// 	}

// 	if (version_flag) {
// 		tlog::none() << "neural graphics primitives version " NGP_VERSION;
// 		return 0;
// 	}

// 	try {
// 		ETestbedMode mode;
// 		if (!mode_flag) {
// 			if (!scene_flag) {
// 				tlog::error() << "Must specify either a mode or a scene";
// 				return 1;
// 			}

// 			fs::path scene_path = get(scene_flag);
// 			if (!scene_path.exists()) {
// 				tlog::error() << "Scene path " << scene_path << " does not exist.";
// 				return 1;
// 			}

// 			if (scene_path.is_directory() || equals_case_insensitive(scene_path.extension(), "json")) {
// 				mode = ETestbedMode::Nerf;
// 			} else if (equals_case_insensitive(scene_path.extension(), "obj") || equals_case_insensitive(scene_path.extension(), "stl")) {
// 				mode = ETestbedMode::Sdf;
// 			} else if (equals_case_insensitive(scene_path.extension(), "nvdb")) {
// 				mode = ETestbedMode::Volume;
// 			} else {
// 				mode = ETestbedMode::Image;
// 			}
// 		} else {
// 			auto mode_str = get(mode_flag);
// 			if (equals_case_insensitive(mode_str, "nerf")) {
// 				mode = ETestbedMode::Nerf;
// 			} else if (equals_case_insensitive(mode_str, "sdf")) {
// 				mode = ETestbedMode::Sdf;
// 			} else if (equals_case_insensitive(mode_str, "image")) {
// 				mode = ETestbedMode::Image;
// 			} else if (equals_case_insensitive(mode_str, "volume")) {
// 				mode = ETestbedMode::Volume;
// 			} else {
// 				tlog::error() << "Mode must be one of 'nerf', 'sdf', 'image', and 'volume'.";
// 				return 1;
// 			}
// 		}

// 		Testbed testbed{mode};

// 		if (scene_flag) {
// 			fs::path scene_path = get(scene_flag);
// 			if (!scene_path.exists()) {
// 				tlog::error() << "Scene path " << scene_path << " does not exist.";
// 				return 1;
// 			}
// 			testbed.load_training_data(scene_path.str());
// 		}

// 		std::string mode_str;
// 		switch (mode) {
// 			case ETestbedMode::Nerf:   mode_str = "nerf";   break;
// 			case ETestbedMode::Sdf:    mode_str = "sdf";    break;
// 			case ETestbedMode::Image:  mode_str = "image";  break;
// 			case ETestbedMode::Volume: mode_str = "volume"; break;
// 		}

// 		if (snapshot_flag) {
// 			// Load network from a snapshot if one is provided
// 			fs::path snapshot_path = get(snapshot_flag);
// 			if (!snapshot_path.exists()) {
// 				tlog::error() << "Snapshot path " << snapshot_path << " does not exist.";
// 				return 1;
// 			}

// 			testbed.load_snapshot(snapshot_path.str());
// 			testbed.m_train = false;
// 		} else {
// 			// Otherwise, load the network config and prepare for training
// 			fs::path network_config_path = fs::path{"configs"}/mode_str;
// 			if (network_config_flag) {
// 				auto network_config_str = get(network_config_flag);
// 				if ((network_config_path/network_config_str).exists()) {
// 					network_config_path = network_config_path/network_config_str;
// 				} else {
// 					network_config_path = network_config_str;
// 				}
// 			} else {
// 				network_config_path = network_config_path/"base.json";
// 			}

// 			if (!network_config_path.exists()) {
// 				tlog::error() << "Network config path " << network_config_path << " does not exist.";
// 				return 1;
// 			}

// 			testbed.reload_network_from_file(network_config_path.str());
// 			testbed.m_train = !no_train_flag;
// 		}

// 		bool gui = !no_gui_flag;
// #ifndef NGP_GUI
// 		gui = false;
// #endif

// 		if (gui) {
// 			testbed.init_window(width_flag ? get(width_flag) : 1920, height_flag ? get(height_flag) : 1080);
// 		}

// 		// Render/training loop
// 		while (testbed.frame()) {
// 			if (!gui) {
// 				tlog::info() << "iteration=" << testbed.m_training_step << " loss=" << testbed.m_loss_scalar.val();
// 			}
// 		}
// 	} catch (const exception& e) {
// 		tlog::error() << "Uncaught exception: " << e.what();
// 		return 1;
// 	}
// }

#include <neural-graphics-primitives/jobs.h>

#include <neural-graphics-primitives/adam_optimizer.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/envmap.cuh>
#include <neural-graphics-primitives/marching_cubes.h>
#include <neural-graphics-primitives/nerf_loader.h>
#include <neural-graphics-primitives/nerf_network.h>
#include <neural-graphics-primitives/render_buffer.h>
// #include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/trainable_buffer.cuh>
#include <neural-graphics-primitives/triangle_octree.cuh>
#include <omp.h>

#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>

#include <filesystem/directory.h>
#include <filesystem/path.h>

#include <zstr.hpp>

#include <fstream>
#include <set>
#include <unordered_set>

using namespace std::literals::chrono_literals;
using namespace tcnn;
namespace fs = filesystem;

NGP_NAMESPACE_BEGIN

float total_time;
inline constexpr __device__ float NERF_RENDERING_NEAR_DISTANCE() { return 0.05f; }
inline constexpr __device__ uint32_t NERF_STEPS() { return 1024; } // finest number of steps per unit length
inline constexpr __device__ uint32_t NERF_CASCADES() { return 8; }

inline constexpr __device__ float SQRT3() { return 1.73205080757f; }
inline constexpr __device__ float STEPSIZE() { return (SQRT3() / NERF_STEPS()); } // for nerf raymarch
inline constexpr __device__ float MIN_CONE_STEPSIZE() { return STEPSIZE(); }
// Maximum step size is the width of the coarsest gridsize cell.
inline constexpr __device__ float MAX_CONE_STEPSIZE() { return STEPSIZE() * (1<<(NERF_CASCADES()-1)) * NERF_STEPS() / NERF_GRIDSIZE(); }

// Used to index into the PRNG stream. Must be larger than the number of
// samples consumed by any given training ray.
inline constexpr __device__ uint32_t N_MAX_RANDOM_SAMPLES_PER_RAY() { return 8; }

// Any alpha below this is considered "invisible" and is thus culled away.
inline constexpr __device__ float NERF_MIN_OPTICAL_THICKNESS() { return 0.01f; }

static constexpr uint32_t MARCH_ITER = 10000;

static constexpr uint32_t MIN_STEPS_INBETWEEN_COMPACTION = 1;
static constexpr uint32_t MAX_STEPS_INBETWEEN_COMPACTION = 16;

void NerfJobs::load_training_data(const fs::path& path) {
	if (!path.exists()) {
		throw std::runtime_error{fmt::format("Data path '{}' does not exist.", path.str())};
	}

	// Automatically determine the mode from the first scene that's loaded
	ETestbedMode scene_mode = mode_from_scene(path.str());
	if (scene_mode == ETestbedMode::None) {
		throw std::runtime_error{fmt::format("Unknown scene format for path '{}'.", path.str())};
	}

	set_mode(scene_mode);

	m_data_path = path;

	switch (m_testbed_mode) {
		case ETestbedMode::Nerf:   load_nerf(path); break;
		case ETestbedMode::Image:  load_image(path); break;
		// case ETestbedMode::Volume: load_volume(path); break;
		default: throw std::runtime_error{"Invalid testbed mode."};
	}

	m_training_data_available = true;

	update_imgui_paths();
}

std::string get_filename_in_data_path_with_suffix_jobs(fs::path data_path, fs::path network_config_path, const char* suffix) {
	// use the network config name along with the data path to build a filename with the requested suffix & extension
	std::string default_name = network_config_path.basename();
	if (default_name == "") {
		default_name = "base";
	}

	if (data_path.empty()) {
		return default_name + std::string(suffix);
	}

	if (data_path.is_directory()) {
		return (data_path / (default_name + std::string{suffix})).str();
	}

	return data_path.stem().str() + "_" + default_name + std::string(suffix);
}

void NerfJobs::update_imgui_paths() {
	snprintf(m_imgui.cam_path_path, sizeof(m_imgui.cam_path_path), "%s", get_filename_in_data_path_with_suffix_jobs(m_data_path, m_network_config_path, "_cam.json").c_str());
	snprintf(m_imgui.extrinsics_path, sizeof(m_imgui.extrinsics_path), "%s", get_filename_in_data_path_with_suffix_jobs(m_data_path, m_network_config_path, "_extrinsics.json").c_str());
	snprintf(m_imgui.mesh_path, sizeof(m_imgui.mesh_path), "%s", get_filename_in_data_path_with_suffix_jobs(m_data_path, m_network_config_path, ".obj").c_str());
	snprintf(m_imgui.snapshot_path, sizeof(m_imgui.snapshot_path), "%s", get_filename_in_data_path_with_suffix_jobs(m_data_path, m_network_config_path, ".ingp").c_str());
	snprintf(m_imgui.video_path, sizeof(m_imgui.video_path), "%s", get_filename_in_data_path_with_suffix_jobs(m_data_path, m_network_config_path, "_video.mp4").c_str());
}

void NerfJobs::Nerf::Training::reset_camera_extrinsics() {
	for (auto&& opt : cam_rot_offset) {
		opt.reset_state();
	}

	for (auto&& opt : cam_pos_offset) {
		opt.reset_state();
	}

	for (auto&& opt : cam_exposure) {
		opt.reset_state();
	}
}

json merge_parent_network_config_jobs(const json& child, const fs::path& child_path) {
	if (!child.contains("parent")) {
		return child;
	}
	fs::path parent_path = child_path.parent_path() / std::string(child["parent"]);
	tlog::info() << "Loading parent network config from: " << parent_path.str();
	std::ifstream f{native_string(parent_path)};
	json parent = json::parse(f, nullptr, true, true);
	parent = merge_parent_network_config_jobs(parent, parent_path);
	parent.merge_patch(child);
	return parent;
}

json NerfJobs::load_network_config(const fs::path& network_config_path) {
	bool is_snapshot = equals_case_insensitive(network_config_path.extension(), "msgpack") || equals_case_insensitive(network_config_path.extension(), "ingp");
	if (network_config_path.empty() || !network_config_path.exists()) {
		throw std::runtime_error{fmt::format("Network {} '{}' does not exist.", is_snapshot ? "snapshot" : "config", network_config_path.str())};
	}

	tlog::info() << "Loading network " << (is_snapshot ? "snapshot" : "config") << " from: " << network_config_path;

	json result;
	if (is_snapshot) {
		std::ifstream f{native_string(network_config_path), std::ios::in | std::ios::binary};
		if (equals_case_insensitive(network_config_path.extension(), "ingp")) {
			// zstr::ifstream applies zlib compression.
			zstr::istream zf{f};
			result = json::from_msgpack(zf);
		} else {
			result = json::from_msgpack(f);
		}
		// we assume parent pointers are already resolved in snapshots.
	} else if (equals_case_insensitive(network_config_path.extension(), "json")) {
		std::ifstream f{native_string(network_config_path)};
		result = json::parse(f, nullptr, true, true);
		result = merge_parent_network_config_jobs(result, network_config_path);
	}

	return result;
}

__global__ void grid_to_bitfield_jobs(
	const uint32_t n_elements,
	const uint32_t n_nonzero_elements,
	const float* __restrict__ grid,
	uint8_t* __restrict__ grid_bitfield,
	const float* __restrict__ mean_density_ptr
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	if (i >= n_nonzero_elements) {
		grid_bitfield[i] = 0;
		return;
	}

	uint8_t bits = 0;

	float thresh = std::min(NERF_MIN_OPTICAL_THICKNESS(), *mean_density_ptr);

	NGP_PRAGMA_UNROLL
	for (uint8_t j = 0; j < 8; ++j) {
		bits |= grid[i*8+j] > thresh ? ((uint8_t)1 << j) : 0;
	}

	grid_bitfield[i] = bits;
}

__global__ void bitfield_max_pool_jobs(const uint32_t n_elements,
	const uint8_t* __restrict__ prev_level,
	uint8_t* __restrict__ next_level
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	uint8_t bits = 0;

	NGP_PRAGMA_UNROLL
	for (uint8_t j = 0; j < 8; ++j) {
		// If any bit is set in the previous level, set this
		// level's bit. (Max pooling.)
		bits |= prev_level[i*8+j] > 0 ? ((uint8_t)1 << j) : 0;
	}

	uint32_t x = tcnn::morton3D_invert(i>>0) + NERF_GRIDSIZE()/8;
	uint32_t y = tcnn::morton3D_invert(i>>1) + NERF_GRIDSIZE()/8;
	uint32_t z = tcnn::morton3D_invert(i>>2) + NERF_GRIDSIZE()/8;

	next_level[tcnn::morton3D(x, y, z)] |= bits;
}

uint8_t* NerfJobs::Nerf::get_density_grid_bitfield_mip(uint32_t mip) {
	return density_grid_bitfield.data() + NERF_GRID_N_CELLS() * ((mip)/8);
}

void NerfJobs::update_density_grid_mean_and_bitfield(cudaStream_t stream) {
	const uint32_t n_elements = NERF_GRID_N_CELLS();

	size_t size_including_mips = NERF_GRID_N_CELLS() * ((NERF_CASCADES())/8);
	m_nerf.density_grid_bitfield.enlarge(size_including_mips);
	m_nerf.density_grid_mean.enlarge(reduce_sum_workspace_size(n_elements));

	CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.density_grid_mean.data(), 0, sizeof(float), stream));
	reduce_sum(m_nerf.density_grid.data(), [n_elements] __device__ (float val) { return fmaxf(val, 0.f) / (n_elements); }, m_nerf.density_grid_mean.data(), n_elements, stream);

	linear_kernel(grid_to_bitfield_jobs, 0, stream, n_elements/8 * NERF_CASCADES(), n_elements/8 * (m_nerf.max_cascade + 1), m_nerf.density_grid.data(), m_nerf.density_grid_bitfield.data(), m_nerf.density_grid_mean.data());

	for (uint32_t level = 1; level < NERF_CASCADES(); ++level) {
		linear_kernel(bitfield_max_pool_jobs, 0, stream, n_elements/64, m_nerf.get_density_grid_bitfield_mip(level-1), m_nerf.get_density_grid_bitfield_mip(level));
	}

	set_all_devices_dirty();
}

void NerfJobs::set_all_devices_dirty() {
	for (auto& device : m_devices) {
		device.set_dirty(true);
	}
}

void NerfJobs::set_max_level(float maxlevel) {
	if (!m_network) return;
	auto hg_enc = dynamic_cast<GridEncoding<network_precision_t>*>(m_encoding.get());
	if (hg_enc) {
		hg_enc->set_max_level(maxlevel);
	}

	reset_accumulation();
}

NerfJobs::NerfJobs(ETestbedMode mode) {
	if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))) {
		throw std::runtime_error{"Testbed requires CUDA 10.2 or later."};
	}

#ifdef NGP_GUI
	// Ensure we're running on the GPU that'll host our GUI. To do so, try creating a dummy
	// OpenGL context, figure out the GPU it's running on, and then kill that context again.
	if (!is_wsl() && glfwInit()) {
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
		GLFWwindow* offscreen_context = glfwCreateWindow(640, 480, "", NULL, NULL);

		if (offscreen_context) {
			glfwMakeContextCurrent(offscreen_context);

			int gl_device = -1;
			unsigned int device_count = 0;
			if (cudaGLGetDevices(&device_count, &gl_device, 1, cudaGLDeviceListAll) == cudaSuccess) {
				if (device_count > 0 && gl_device >= 0) {
					set_cuda_device(gl_device);
				}
			}

			glfwDestroyWindow(offscreen_context);
		}

		glfwTerminate();
	}
#endif

	// Reset our stream, which was allocated on the originally active device,
	// to make sure it corresponds to the now active device.
	m_stream = {};

	int active_device = cuda_device();
	int active_compute_capability = cuda_compute_capability();
	tlog::success() << "Initialized CUDA. Active GPU is #" << active_device << ": " << cuda_device_name() << " [" << active_compute_capability << "]";

	if (active_compute_capability < MIN_GPU_ARCH) {
		tlog::warning() << "Insufficient compute capability " << active_compute_capability << " detected.";
		tlog::warning() << "This program was compiled for >=" << MIN_GPU_ARCH << " and may thus behave unexpectedly.";
	}

	m_devices.emplace_back(active_device, true);
	
	// Multi-GPU is only supported in NeRF mode for now
	int n_devices = 8;//cuda_device_count();
	for (int i = 0; i < n_devices; ++i) {
		if (i == active_device) {
			continue;
		}
		if (cuda_compute_capability(0) >= MIN_GPU_ARCH) {
			
			m_devices.emplace_back(0, false);
		}
	}

	if (m_devices.size() > 1) {
		tlog::success() << "Detected auxiliary GPUs:";
		for (size_t i = 1; i < m_devices.size(); ++i) {
			const auto& device = m_devices[i];
			tlog::success() << "  #" << device.id() << ": " << device.name() << " [" << device.compute_capability() << "]";
		}
	}

	m_network_config = {
		{"loss", {
			{"otype", "L2"}
		}},
		{"optimizer", {
			{"otype", "Adam"},
			{"learning_rate", 1e-3},
			{"beta1", 0.9f},
			{"beta2", 0.99f},
			{"epsilon", 1e-15f},
			{"l2_reg", 1e-6f},
		}},
		{"encoding", {
			{"otype", "HashGrid"},
			{"n_levels", 16},
			{"n_features_per_level", 2},
			{"log2_hashmap_size", 19},
			{"base_resolution", 16},
		}},
		{"network", {
			{"otype", "FullyFusedMLP"},
			{"n_neurons", 64},
			{"n_layers", 2},
			{"activation", "ReLU"},
			{"output_activation", "None"},
		}},
	};

	set_mode(mode);
	set_exposure(0);
	set_max_level(1.f);

	reset_camera();
}

void NerfJobs::destroy_window() {
#ifndef NGP_GUI
	throw std::runtime_error{"destroy_window failed: NGP was built without GUI support"};
#endif
}

__host__ __device__ vec3 warp_direction_jobs(const vec3& dir) {
	return (dir + vec3(1.0f)) * 0.5f;
}

void NerfJobs::Nerf::Training::update_extra_dims() {
	uint32_t n_extra_dims = dataset.n_extra_dims();
	std::vector<float> extra_dims_cpu(extra_dims_gpu.size());
	for (uint32_t i = 0; i < extra_dims_opt.size(); ++i) {
		const std::vector<float>& value = extra_dims_opt[i].variable();
		for (uint32_t j = 0; j < n_extra_dims; ++j) {
			extra_dims_cpu[i * n_extra_dims + j] = value[j];
		}
	}

	CUDA_CHECK_THROW(cudaMemcpyAsync(extra_dims_gpu.data(), extra_dims_cpu.data(), extra_dims_opt.size() * n_extra_dims * sizeof(float), cudaMemcpyHostToDevice));
}

void NerfJobs::Nerf::Training::update_transforms(int first, int last) {
	if (last < 0) {
		last = dataset.n_images;
	}

	if (last > dataset.n_images) {
		last = dataset.n_images;
	}

	int n = last - first;
	if (n <= 0) {
		return;
	}

	if (transforms.size() < last) {
		transforms.resize(last);
	}

	for (uint32_t i = 0; i < n; ++i) {
		auto xform = dataset.xforms[i + first];
		float det_start = determinant(mat3(xform.start));
		float det_end = determinant(mat3(xform.end));
		if (distance(det_start, 1.0f) > 0.01f || distance(det_end, 1.0f) > 0.01f) {
			tlog::warning() << "Rotation of camera matrix in frame " << i + first << " has a scaling component (determinant!=1).";
			tlog::warning() << "Normalizing the matrix. This hints at an issue in your data generation pipeline and should be fixed.";

			xform.start[0] /= std::cbrt(det_start); xform.start[1] /= std::cbrt(det_start); xform.start[2] /= std::cbrt(det_start);
			xform.end[0]   /= std::cbrt(det_end);   xform.end[1]   /= std::cbrt(det_end);   xform.end[2]   /= std::cbrt(det_end);
			dataset.xforms[i + first] = xform;
		}

		mat3 rot = rotmat(cam_rot_offset[i + first].variable());
		auto rot_start = rot * mat3(xform.start);
		auto rot_end = rot * mat3(xform.end);
		xform.start = mat4x3(rot_start[0], rot_start[1], rot_start[2], xform.start[3]);
		xform.end = mat4x3(rot_end[0], rot_end[1], rot_end[2], xform.end[3]);

		xform.start[3] += cam_pos_offset[i + first].variable();
		xform.end[3] += cam_pos_offset[i + first].variable();
		transforms[i + first] = xform;
	}

	transforms_gpu.enlarge(last);
	CUDA_CHECK_THROW(cudaMemcpy(transforms_gpu.data() + first, transforms.data() + first, n * sizeof(TrainingXForm), cudaMemcpyHostToDevice));
}

bool NerfJobs::clear_tmp_dir() {
	wait_all(m_render_futures);
	m_render_futures.clear();

	bool success = true;
	auto tmp_dir = fs::path{"tmp"};
	if (tmp_dir.exists()) {
		if (tmp_dir.is_directory()) {
			for (const auto& path : fs::directory{tmp_dir}) {
				if (path.is_file()) {
					success &= path.remove_file();
				}
			}
		}

		success &= tmp_dir.remove_file();
	}

	return success;
}

NerfJobs::~NerfJobs() {

	// If any temporary file was created, make sure it's deleted
	clear_tmp_dir();

	if (m_render_window) {
		destroy_window();
	}
}

void NerfJobs::set_camera_to_training_view(int trainview) {
	auto old_look_at = look_at();
	m_camera = m_smoothed_camera = get_xform_given_rolling_shutter(m_nerf.training.transforms[trainview], m_nerf.training.dataset.metadata[trainview].rolling_shutter, vec2{0.5f, 0.5f}, 0.0f);
	m_relative_focal_length = m_nerf.training.dataset.metadata[trainview].focal_length / (float)m_nerf.training.dataset.metadata[trainview].resolution[m_fov_axis];
	m_scale = std::max(dot(old_look_at - view_pos(), view_dir()), 0.1f);
	m_nerf.render_with_lens_distortion = true;
	m_nerf.render_lens = m_nerf.training.dataset.metadata[trainview].lens;
	if (!supports_dlss(m_nerf.render_lens.mode)) {
		m_dlss = false;
	}

	m_screen_center = vec2(1.0f) - m_nerf.training.dataset.metadata[trainview].principal_point;
	m_nerf.training.view = trainview;

	reset_accumulation(true);
}

void NerfJobs::sync_device(CudaRenderBuffer& render_buffer, NerfJobs::CudaDevice& device) {
	if (!device.dirty()) {
		return;
	}

	if (device.is_primary()) {
		device.data().density_grid_bitfield_ptr = m_nerf.density_grid_bitfield.data();
		device.data().hidden_area_mask = render_buffer.hidden_area_mask();
		device.set_dirty(false);
		return;
	}

	m_stream.signal(device.stream());

	int active_device = cuda_device();
	auto guard = device.device_guard();

	device.data().density_grid_bitfield.resize(m_nerf.density_grid_bitfield.size());
	if (m_nerf.density_grid_bitfield.size() > 0) {
		CUDA_CHECK_THROW(cudaMemcpyPeerAsync(device.data().density_grid_bitfield.data(), device.id(), m_nerf.density_grid_bitfield.data(), active_device, m_nerf.density_grid_bitfield.bytes(), device.stream()));
	}

	device.data().density_grid_bitfield_ptr = device.data().density_grid_bitfield.data();

	if (m_network) {
		device.data().params.resize(m_network->n_params());
		CUDA_CHECK_THROW(cudaMemcpyPeerAsync(device.data().params.data(), device.id(), m_network->inference_params(), active_device, device.data().params.bytes(), device.stream()));
		device.nerf_network()->set_params(device.data().params.data(), device.data().params.data(), nullptr);
	}

	if (render_buffer.hidden_area_mask()) {
		auto ham = std::make_shared<Buffer2D<uint8_t>>(render_buffer.hidden_area_mask()->resolution());
		CUDA_CHECK_THROW(cudaMemcpyPeerAsync(ham->data(), device.id(), render_buffer.hidden_area_mask()->data(), active_device, ham->bytes(), device.stream()));
		device.data().hidden_area_mask = ham;
	} else {
		device.data().hidden_area_mask = nullptr;
	}

	device.set_dirty(false);
}

template <class F>
auto make_copyable_function(F&& f) {
	using dF = std::decay_t<F>;
	auto spf = std::make_shared<dF>(std::forward<F>(f));
	return [spf](auto&&... args) -> decltype(auto) {
		return (*spf)( decltype(args)(args)... );
	};
}

ScopeGuard NerfJobs::use_device(cudaStream_t stream, CudaRenderBuffer& render_buffer, NerfJobs::CudaDevice& device) {

	// device.wait_for(stream);
	if (device.is_primary()) {
		device.set_render_buffer_view(render_buffer.view());
		return ScopeGuard{[&device, stream]() {
			device.set_render_buffer_view({});
			device.signal(stream);
		}};
	}
	
	int active_device = cuda_device();
	auto guard = device.device_guard();

	size_t n_pixels = compMul(render_buffer.in_resolution());
	
	GPUMemoryArena::Allocation alloc;
	auto scratch = allocate_workspace_and_distribute<vec4, float>(device.stream(), &alloc, n_pixels, n_pixels);

	device.set_render_buffer_view({
		std::get<0>(scratch),
		std::get<1>(scratch),
		render_buffer.in_resolution(),
		render_buffer.spp(),
		device.data().hidden_area_mask,
	});

	return ScopeGuard{make_copyable_function([&render_buffer, &device, guard=std::move(guard), alloc=std::move(alloc), active_device, stream]() {
		// Copy device's render buffer's data onto the original render buffer
		CUDA_CHECK_THROW(cudaMemcpyPeerAsync(render_buffer.frame_buffer(), active_device, device.render_buffer_view().frame_buffer, device.id(), compMul(render_buffer.in_resolution()) * sizeof(vec4), device.stream()));
		CUDA_CHECK_THROW(cudaMemcpyPeerAsync(render_buffer.depth_buffer(), active_device, device.render_buffer_view().depth_buffer, device.id(), compMul(render_buffer.in_resolution()) * sizeof(float), device.stream()));

		device.set_render_buffer_view({});
		device.signal(stream);
	})};
}

void NerfJobs::NerfTracer::enlarge(size_t n_elements, uint32_t padded_output_width, uint32_t n_extra_dims, cudaStream_t stream) {
	n_elements = next_multiple(n_elements, size_t(tcnn::batch_size_granularity));
	size_t num_floats = sizeof(NerfCoordinate) / 4 + n_extra_dims;
	auto scratch = allocate_workspace_and_distribute<
		vec4, float, NerfPayload, // m_rays[0]
		vec4, float, NerfPayload, // m_rays[1]
		vec4, float, NerfPayload, // m_rays_hit

		network_precision_t,
		float,
		uint32_t,
		uint32_t
	>(
		stream, &m_scratch_alloc,
		n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements,
		n_elements * MAX_STEPS_INBETWEEN_COMPACTION * padded_output_width,
		n_elements * MAX_STEPS_INBETWEEN_COMPACTION * num_floats,
		32, // 2 full cache lines to ensure no overlap
		32  // 2 full cache lines to ensure no overlap
	);

	m_rays[0].set(std::get<0>(scratch), std::get<1>(scratch), std::get<2>(scratch), n_elements);
	m_rays[1].set(std::get<3>(scratch), std::get<4>(scratch), std::get<5>(scratch), n_elements);
	m_rays_hit.set(std::get<6>(scratch), std::get<7>(scratch), std::get<8>(scratch), n_elements);

	m_network_output = std::get<9>(scratch);
	m_network_input = std::get<10>(scratch);

	m_hit_counter = std::get<11>(scratch);
	m_alive_counter = std::get<12>(scratch);
}

__global__ void init_rays_with_payload_kernel_jobs(
	uint32_t sample_index,
	NerfPayload* __restrict__ payloads,
	ivec2 resolution,
	vec2 focal_length,
	mat4x3 camera_matrix0,
	mat4x3 camera_matrix1,
	vec4 rolling_shutter,
	vec2 screen_center,
	vec3 parallax_shift,
	bool snap_to_pixel_centers,
	BoundingBox render_aabb,
	mat3 render_aabb_to_local,
	float near_distance,
	float plane_z,
	float aperture_size,
	Foveation foveation,
	Lens lens,
	Buffer2DView<const vec4> envmap,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer,
	Buffer2DView<const uint8_t> hidden_area_mask,
	Buffer2DView<const vec2> distortion,
	ERenderMode render_mode
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	uint32_t idx = x + resolution.x * y;

	if (plane_z < 0) {
		aperture_size = 0.0;
	}

	vec2 pixel_offset = ld_random_pixel_offset(snap_to_pixel_centers ? 0 : sample_index);
	vec2 uv = vec2{(float)x + pixel_offset.x, (float)y + pixel_offset.y} / vec2(resolution);
	float ray_time = rolling_shutter.x + rolling_shutter.y * uv.x + rolling_shutter.z * uv.y + rolling_shutter.w * ld_random_val(sample_index, idx * 72239731);
	Ray ray = uv_to_ray(
		sample_index,
		uv,
		resolution,
		focal_length,
		camera_matrix0 * ray_time + camera_matrix1 * (1.f - ray_time),
		screen_center,
		parallax_shift,
		near_distance,
		plane_z,
		aperture_size,
		foveation,
		hidden_area_mask,
		lens,
		distortion
	);

	NerfPayload& payload = payloads[idx];
	payload.max_weight = 0.0f;

	depth_buffer[idx] = MAX_DEPTH();

	if (!ray.is_valid()) {
		payload.origin = ray.o;
		payload.alive = false;
		return;
	}

	if (plane_z < 0) {
		float n = length(ray.d);
		payload.origin = ray.o;
		payload.dir = (1.0f/n) * ray.d;
		payload.t = -plane_z*n;
		payload.idx = idx;
		payload.n_steps = 0;
		payload.alive = false;
		depth_buffer[idx] = -plane_z;
		return;
	}

	ray.d = normalize(ray.d);

	if (envmap) {
		frame_buffer[idx] = read_envmap(envmap, ray.d);
	}

	float t = fmaxf(render_aabb.ray_intersect(render_aabb_to_local * ray.o, render_aabb_to_local * ray.d).x, 0.0f) + 1e-6f;

	if (!render_aabb.contains(render_aabb_to_local * ray(t))) {
		payload.origin = ray.o;
		payload.alive = false;
		return;
	}

	if (render_mode == ERenderMode::Distortion) {
		vec2 offset = vec2(0.0f);
		if (distortion) {
			offset += distortion.at_lerp(vec2{(float)x + 0.5f, (float)y + 0.5f} / vec2(resolution));
		}

		frame_buffer[idx].rgb() = to_rgb(offset * 50.0f);
		frame_buffer[idx].a = 1.0f;
		depth_buffer[idx] = 1.0f;
		payload.origin = ray(MAX_DEPTH());
		payload.alive = false;
		return;
	}

	payload.origin = ray.o;
	payload.dir = ray.d;
	payload.t = t;
	payload.idx = idx;
	payload.n_steps = 0;
	payload.alive = true;
}

inline __host__ __device__ uint32_t grid_mip_offset_jobs(uint32_t mip) {
	return (NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE()) * mip;
}

inline __device__ int mip_from_pos_jobs(const vec3& pos, uint32_t max_cascade = NERF_CASCADES()-1) {
	int exponent;
	float maxval = compMax(abs(pos - vec3(0.5f)));
	frexpf(maxval, &exponent);
	return (uint32_t)tcnn::clamp(exponent+1, 0, (int)max_cascade);
}

__device__ uint32_t cascaded_grid_idx_at_jobs(vec3 pos, uint32_t mip) {
	float mip_scale = scalbnf(1.0f, -mip);
	pos -= vec3(0.5f);
	pos *= mip_scale;
	pos += vec3(0.5f);

	ivec3 i = pos * (float)NERF_GRIDSIZE();
	if (i.x < 0 || i.x >= NERF_GRIDSIZE() || i.y < 0 || i.y >= NERF_GRIDSIZE() || i.z < 0 || i.z >= NERF_GRIDSIZE()) {
		return 0xFFFFFFFF;
	}

	return tcnn::morton3D(i.x, i.y, i.z);
}

inline __device__ int mip_from_dt_jobs(float dt, const vec3& pos, uint32_t max_cascade = NERF_CASCADES()-1) {
	uint32_t mip = mip_from_pos_jobs(pos, max_cascade);
	dt *= 2 * NERF_GRIDSIZE();
	if (dt < 1.0f) {
		return mip;
	}

	int exponent;
	frexpf(dt, &exponent);
	return (uint32_t)tcnn::clamp((int)mip, exponent, (int)max_cascade);
}

__device__ bool density_grid_occupied_at_jobs(const vec3& pos, const uint8_t* density_grid_bitfield, uint32_t mip) {
	uint32_t idx = cascaded_grid_idx_at_jobs(pos, mip);
	if (idx == 0xFFFFFFFF) {
		return false;
	}
	return density_grid_bitfield[idx/8+grid_mip_offset_jobs(mip)/8] & (1<<(idx%8));
}


__device__ float cascaded_grid_at_jobs(vec3 pos, const float* cascaded_grid, uint32_t mip) {
	uint32_t idx = cascaded_grid_idx_at_jobs(pos, mip);
	if (idx == 0xFFFFFFFF) {
		return 0.0f;
	}
	return cascaded_grid[idx+grid_mip_offset_jobs(mip)];
}

__device__ float& cascaded_grid_at_jobs(vec3 pos, float* cascaded_grid, uint32_t mip) {
	uint32_t idx = cascaded_grid_idx_at_jobs(pos, mip);
	if (idx == 0xFFFFFFFF) {
		idx = 0;
		printf("WARNING: invalid cascaded grid access.");
	}
	return cascaded_grid[idx+grid_mip_offset_jobs(mip)];
}

inline __device__ float distance_to_next_voxel_jobs(const vec3& pos, const vec3& dir, const vec3& idir, float res) { // dda like step
	vec3 p = res * (pos - vec3(0.5f));
	float tx = (floorf(p.x + 0.5f + 0.5f * sign(dir.x)) - p.x) * idir.x;
	float ty = (floorf(p.y + 0.5f + 0.5f * sign(dir.y)) - p.y) * idir.y;
	float tz = (floorf(p.z + 0.5f + 0.5f * sign(dir.z)) - p.z) * idir.z;
	float t = min(min(tx, ty), tz);

	return fmaxf(t / res, 0.0f);
}

inline __host__ __device__ float to_stepping_space_jobs(float t, float cone_angle) {
	if (cone_angle <= 1e-5f) {
		return t / MIN_CONE_STEPSIZE();
	}

	float log1p_c = logf(1.0f + cone_angle);

	float a = (logf(MIN_CONE_STEPSIZE()) - logf(log1p_c)) / log1p_c;
	float b = (logf(MAX_CONE_STEPSIZE()) - logf(log1p_c)) / log1p_c;

	float at = expf(a * log1p_c);
	float bt = expf(b * log1p_c);

	if (t <= at) {
		return (t - at) / MIN_CONE_STEPSIZE() + a;
	} else if (t <= bt) {
		return logf(t) / log1p_c;
	} else {
		return (t - bt) / MAX_CONE_STEPSIZE() + b;
	}
}

inline __host__ __device__ float from_stepping_space_jobs(float n, float cone_angle) {
	if (cone_angle <= 1e-5f) {
		return n * MIN_CONE_STEPSIZE();
	}

	float log1p_c = logf(1.0f + cone_angle);

	float a = (logf(MIN_CONE_STEPSIZE()) - logf(log1p_c)) / log1p_c;
	float b = (logf(MAX_CONE_STEPSIZE()) - logf(log1p_c)) / log1p_c;

	float at = expf(a * log1p_c);
	float bt = expf(b * log1p_c);

	if (n <= a) {
		return (n - a) * MIN_CONE_STEPSIZE() + at;
	} else if (n <= b) {
		return expf(n * log1p_c);
	} else {
		return (n - b) * MAX_CONE_STEPSIZE() + bt;
	}
}

inline __device__ float advance_to_next_voxel_jobs(float t, float cone_angle, const vec3& pos, const vec3& dir, const vec3& idir, uint32_t mip) {
	float res = scalbnf(NERF_GRIDSIZE(), -(int)mip);

	float t_target = t + distance_to_next_voxel_jobs(pos, dir, idir, res);

	// Analytic stepping in multiples of 1 in the "log-space" of our exponential stepping routine
	t = to_stepping_space_jobs(t, cone_angle);
	t_target = to_stepping_space_jobs(t_target, cone_angle);

	return from_stepping_space_jobs(t + ceilf(fmaxf(t_target - t, 0.5f)), cone_angle);
}


// inline __host__ __device__ float from_stepping_space_jobs(float n, float cone_angle) {
// 	if (cone_angle <= 1e-5f) {
// 		return n * MIN_CONE_STEPSIZE();
// 	}

// 	float log1p_c = logf(1.0f + cone_angle);

// 	float a = (logf(MIN_CONE_STEPSIZE()) - logf(log1p_c)) / log1p_c;
// 	float b = (logf(MAX_CONE_STEPSIZE()) - logf(log1p_c)) / log1p_c;

// 	float at = expf(a * log1p_c);
// 	float bt = expf(b * log1p_c);

// 	if (n <= a) {
// 		return (n - a) * MIN_CONE_STEPSIZE() + at;
// 	} else if (n <= b) {
// 		return expf(n * log1p_c);
// 	} else {
// 		return (n - b) * MAX_CONE_STEPSIZE() + bt;
// 	}
// }

inline __host__ __device__ float advance_n_steps_jobs(float t, float cone_angle, float n) {
	return from_stepping_space_jobs(to_stepping_space_jobs(t, cone_angle) + n, cone_angle);
}

template <bool MIP_FROM_DT=false>
__device__ float if_unoccupied_advance_to_next_occupied_voxel_jobs(
	float t,
	float cone_angle,
	const Ray& ray,
	const vec3& idir,
	const uint8_t* __restrict__ density_grid,
	uint32_t min_mip,
	uint32_t max_mip,
	BoundingBox aabb,
	mat3 aabb_to_local = mat3(1.0f)
) {
	while (true) {
		vec3 pos = ray(t);
		if (t >= MAX_DEPTH() || !aabb.contains(aabb_to_local * pos)) {
			return MAX_DEPTH();
		}

		uint32_t mip = tcnn::clamp((uint32_t)(MIP_FROM_DT ? mip_from_dt_jobs(advance_n_steps_jobs(t, cone_angle, 1.0f) - t, pos) : mip_from_pos_jobs(pos)), min_mip, max_mip);

		if (!density_grid || density_grid_occupied_at_jobs(pos, density_grid, mip)) {
			return t;
		}

		// Find largest empty voxel surrounding us, such that we can advance as far as possible in the next step.
		// Other places that do voxel stepping don't need this, because they don't rely on thread coherence as
		// much as this one here.
		while (mip < max_mip && !density_grid_occupied_at_jobs(pos, density_grid, mip+1)) {
			++mip;
		}

		t = advance_to_next_voxel_jobs(t, cone_angle, pos, ray.d, idir, mip);
	}
}



__device__ vec3 unwarp_position_jobs(const vec3& pos, const BoundingBox& aabb) {
	// return {logit(pos.x) + 0.5f, logit(pos.y) + 0.5f, logit(pos.z) + 0.5f};
	// return pos;

	return aabb.min + pos * aabb.diag();
}

__device__ vec3 unwarp_position_derivative_jobs(const vec3& pos, const BoundingBox& aabb) {
	// return {logit(pos.x) + 0.5f, logit(pos.y) + 0.5f, logit(pos.z) + 0.5f};
	// return pos;

	return aabb.diag();
}

__device__ float unwarp_dt_jobs(float dt) {
	float max_stepsize = MIN_CONE_STEPSIZE() * (1<<(NERF_CASCADES()-1));
	return dt * (max_stepsize - MIN_CONE_STEPSIZE()) + MIN_CONE_STEPSIZE();
}

__device__ float network_to_density_jobs(float val, ENerfActivation activation) {
	switch (activation) {
		case ENerfActivation::None: return val;
		case ENerfActivation::ReLU: return val > 0.0f ? val : 0.0f;
		case ENerfActivation::Logistic: return tcnn::logistic(val);
		case ENerfActivation::Exponential: return __expf(val);
		default: assert(false);
	}
	return 0.0f;
}

__device__ float network_to_rgb_jobs(float val, ENerfActivation activation) {
	switch (activation) {
		case ENerfActivation::None: return val;
		case ENerfActivation::ReLU: return val > 0.0f ? val : 0.0f;
		case ENerfActivation::Logistic: return tcnn::logistic(val);
		case ENerfActivation::Exponential: return __expf(tcnn::clamp(val, -10.0f, 10.0f));
		default: assert(false);
	}
	return 0.0f;
}

template <typename T>
__device__ vec3 network_to_rgb_vec_jobs(const T& val, ENerfActivation activation) {
	return {
		network_to_rgb_jobs(float(val[0]), activation),
		network_to_rgb_jobs(float(val[1]), activation),
		network_to_rgb_jobs(float(val[2]), activation),
	};
}

__global__ void composite_kernel_nerf_jobs(
	const uint32_t n_elements,
	const uint32_t stride,
	const uint32_t current_step,
	BoundingBox aabb,
	float glow_y_cutoff,
	int glow_mode,
	mat4x3 camera_matrix,
	vec2 focal_length,
	float depth_scale,
	vec4* __restrict__ rgba,
	float* __restrict__ depth,
	NerfPayload* payloads,
	PitchedPtr<NerfCoordinate> network_input,
	const tcnn::network_precision_t* __restrict__ network_output,
	uint32_t padded_output_width,
	uint32_t n_steps,
	ERenderMode render_mode,
	const uint8_t* __restrict__ density_grid,
	ENerfActivation rgb_activation,
	ENerfActivation density_activation,
	int show_accel,
	float min_transmittance
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (i >= n_elements) return;

	NerfPayload& payload = payloads[i];

	if (!payload.alive) {
		return;
	}
	vec4 local_rgba = rgba[i];
	float local_depth = depth[i];
	vec3 origin = payload.origin;
	vec3 cam_fwd = camera_matrix[2];
	// Composite in the last n steps
	uint32_t actual_n_steps = payload.n_steps;

	uint32_t j = 0;
	for (; j < actual_n_steps; ++j) {
		tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output;
		local_network_output[0] = network_output[i + j * n_elements + 0 * stride];
		local_network_output[1] = network_output[i + j * n_elements + 1 * stride];
		local_network_output[2] = network_output[i + j * n_elements + 2 * stride];
		local_network_output[3] = network_output[i + j * n_elements + 3 * stride];
		const NerfCoordinate* input = network_input(i + j * n_elements);
		vec3 warped_pos = input->pos.p;
		vec3 pos = unwarp_position_jobs(warped_pos, aabb);

		float T = 1.f - local_rgba.a;
		float dt = unwarp_dt_jobs(input->dt);
		float alpha = 1.f - __expf(-network_to_density_jobs(float(local_network_output[3]), density_activation) * dt);
		if (show_accel >= 0) {
			alpha = 1.f;
		}
		float weight = alpha * T;

		vec3 rgb = network_to_rgb_vec_jobs(local_network_output, rgb_activation);

		if (glow_mode) { // random grid visualizations ftw!
#if 0
			if (0) {  // extremely startrek edition
				float glow_y = (pos.y - (glow_y_cutoff - 0.5f)) * 2.f;
				if (glow_y>1.f) glow_y=max(0.f,21.f-glow_y*20.f);
				if (glow_y>0.f) {
					float line;
					line =max(0.f,cosf(pos.y*2.f*3.141592653589793f * 16.f)-0.95f);
					line+=max(0.f,cosf(pos.x*2.f*3.141592653589793f * 16.f)-0.95f);
					line+=max(0.f,cosf(pos.z*2.f*3.141592653589793f * 16.f)-0.95f);
					line+=max(0.f,cosf(pos.y*4.f*3.141592653589793f * 16.f)-0.975f);
					line+=max(0.f,cosf(pos.x*4.f*3.141592653589793f * 16.f)-0.975f);
					line+=max(0.f,cosf(pos.z*4.f*3.141592653589793f * 16.f)-0.975f);
					glow_y=glow_y*glow_y*0.5f + glow_y*line*25.f;
					rgb.y+=glow_y;
					rgb.z+=glow_y*0.5f;
					rgb.x+=glow_y*0.25f;
				}
			}
#endif
			float glow = 0.f;

			bool green_grid = glow_mode & 1;
			bool green_cutline = glow_mode & 2;
			bool mask_to_alpha = glow_mode & 4;

			// less used?
			bool radial_mode = glow_mode & 8;
			bool grid_mode = glow_mode & 16; // makes object rgb go black!

			{
				float dist;
				if (radial_mode) {
					dist = distance(pos, camera_matrix[3]);
					dist = min(dist, (4.5f - pos.y) * 0.333f);
				} else {
					dist = pos.y;
				}

				if (grid_mode) {
					glow = 1.f / max(1.f, dist);
				} else {
					float y = glow_y_cutoff - dist; // - (ii*0.005f);
					float mask = 0.f;
					if (y > 0.f) {
						y *= 80.f;
						mask = min(1.f, y);
						//if (mask_mode) {
						//	rgb.x=rgb.y=rgb.z=mask; // mask mode
						//} else
						{
							if (green_cutline) {
								glow += max(0.f, 1.f - abs(1.f -y)) * 4.f;
							}

							if (y>1.f) {
								y = 1.f - (y - 1.f) * 0.05f;
							}

							if (green_grid) {
								glow += max(0.f, y / max(1.f, dist));
							}
						}
					}
					if (mask_to_alpha) {
						weight *= mask;
					}
				}
			}

			if (glow > 0.f) {
				float line;
				line  = max(0.f, cosf(pos.y * 2.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.x * 2.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.z * 2.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.y * 4.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.x * 4.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.z * 4.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.y * 8.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.x * 8.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.z * 8.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.y * 16.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.x * 16.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.z * 16.f * 3.141592653589793f * 16.f) - 0.975f);
				if (grid_mode) {
					glow = /*glow*glow*0.75f + */ glow * line * 15.f;
					rgb.y = glow;
					rgb.z = glow * 0.5f;
					rgb.x = glow * 0.25f;
				} else {
					glow = glow * glow * 0.25f + glow * line * 15.f;
					rgb.y += glow;
					rgb.z += glow * 0.5f;
					rgb.x += glow * 0.25f;
				}
			}
		} // glow

		if (render_mode == ERenderMode::Positions) {
			if (show_accel >= 0) {
				uint32_t mip = max(show_accel, mip_from_pos_jobs(pos));
				uint32_t res = NERF_GRIDSIZE() >> mip;
				int ix = pos.x * res;
				int iy = pos.y * res;
				int iz = pos.z * res;
				default_rng_t rng(ix + iy * 232323 + iz * 727272);
				rgb.x = 1.f - mip * (1.f / (NERF_CASCADES() - 1));
				rgb.y = rng.next_float();
				rgb.z = rng.next_float();
			} else {
				rgb = (pos - vec3(0.5f)) / 2.0f + vec3(0.5f);
			}
		} else if (render_mode == ERenderMode::EncodingVis) {
			rgb = warped_pos;
		} else if (render_mode == ERenderMode::Depth) {
			rgb = vec3(dot(cam_fwd, pos - origin) * depth_scale);
		} else if (render_mode == ERenderMode::AO) {
			rgb = vec3(alpha);
		}

		local_rgba += vec4(rgb * weight, weight);
		if (weight > payload.max_weight) {
			payload.max_weight = weight;
			local_depth = dot(cam_fwd, pos - camera_matrix[3]);
		}

		if (local_rgba.a > (1.0f - min_transmittance)) {
			local_rgba /= local_rgba.a;
			break;
		}
	}

	if (j < n_steps) {
		payload.alive = false;
		payload.n_steps = j + current_step;
	}

	rgba[i] = local_rgba;
	depth[i] = local_depth;
	
}


__device__ void advance_pos_jobs(
	NerfPayload& payload,
	const BoundingBox& render_aabb,
	const mat3& render_aabb_to_local,
	const vec3& camera_fwd,
	const vec2& focal_length,
	uint32_t sample_index,
	const uint8_t* __restrict__ density_grid,
	uint32_t min_mip,
	uint32_t max_mip,
	float cone_angle_constant
) {
	if (!payload.alive) {
		return;
	}

	vec3 origin = payload.origin;
	vec3 dir = payload.dir;
	vec3 idir = vec3(1.0f) / dir;

	float cone_angle = cone_angle_constant;

	float t = advance_n_steps_jobs(payload.t, cone_angle, ld_random_val(sample_index, payload.idx * 786433));
	t = if_unoccupied_advance_to_next_occupied_voxel_jobs(t, cone_angle, {origin, dir}, idir, density_grid, min_mip, max_mip, render_aabb, render_aabb_to_local);
	if (t >= MAX_DEPTH()) {
		payload.alive = false;
	} else {
		payload.t = t;
	}
}

__global__ void shade_kernel_nerf_jobs(
	const uint32_t n_elements,
	vec4* __restrict__ rgba,
	float* __restrict__ depth,
	NerfPayload* __restrict__ payloads,
	ERenderMode render_mode,
	bool train_in_linear_colors,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	NerfPayload& payload = payloads[i];

	vec4 tmp = rgba[i];
	if (render_mode == ERenderMode::Normals) {	
		vec3 n = normalize(tmp.xyz());
		tmp.rgb = (0.5f * n + vec3(0.5f)) * tmp.a;
	} else if (render_mode == ERenderMode::Cost) {
		float col = (float)payload.n_steps / 128;
		tmp = {col, col, col, 1.0f};
	}
	if (!train_in_linear_colors && (render_mode == ERenderMode::Shade || render_mode == ERenderMode::Slice)) {
		// Accumulate in linear colors
		tmp.rgb = srgb_to_linear(tmp.rgb);
	}

	frame_buffer[payload.idx] = tmp + frame_buffer[payload.idx] * (1.0f - tmp.a);
	if (render_mode != ERenderMode::Slice && tmp.a > 0.2f) {
		depth_buffer[payload.idx] = depth[i];
	}
}

__global__ void advance_pos_nerf_jobs(
	const uint32_t n_elements,
	BoundingBox render_aabb,
	mat3 render_aabb_to_local,
	vec3 camera_fwd,
	vec2 focal_length,
	uint32_t sample_index,
	NerfPayload* __restrict__ payloads,
	const uint8_t* __restrict__ density_grid,
	uint32_t min_mip,
	uint32_t max_mip,
	float cone_angle_constant
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	advance_pos_jobs(payloads[i], render_aabb, render_aabb_to_local, camera_fwd, focal_length, sample_index, density_grid, min_mip, max_mip, cone_angle_constant);
}


__global__ void compact_kernel_nerf_jobs(
	const uint32_t n_elements,
	vec4* src_rgba, float* src_depth, NerfPayload* src_payloads,
	vec4* dst_rgba, float* dst_depth, NerfPayload* dst_payloads,
	vec4* dst_final_rgba, float* dst_final_depth, NerfPayload* dst_final_payloads,
	uint32_t* counter, uint32_t* finalCounter
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	NerfPayload& src_payload = src_payloads[i];
	
	if (src_payload.alive) {
		uint32_t idx = atomicAdd(counter, 1);
		dst_payloads[idx] = src_payload;
		dst_rgba[idx] = src_rgba[i];
		dst_depth[idx] = src_depth[i];
	} else if (src_rgba[i].a > 0.001f) {
		uint32_t idx = atomicAdd(finalCounter, 1);
		dst_final_payloads[idx] = src_payload;
		dst_final_rgba[idx] = src_rgba[i];
		dst_final_depth[idx] = src_depth[i];
	}
}


void NerfJobs::NerfTracer::enlarge_multi_devices(size_t n_elements, uint32_t padded_output_width, uint32_t n_extra_dims, RaysNerfSoa* m_rays,RaysNerfSoa &m_rays_hit,cudaStream_t stream) {
	n_elements = next_multiple(n_elements, size_t(tcnn::batch_size_granularity));
	size_t num_floats = sizeof(NerfCoordinate) / 4 + n_extra_dims;
	auto scratch = allocate_workspace_and_distribute<
		vec4, float, NerfPayload, // m_rays[0]
		vec4, float, NerfPayload, // m_rays[1]
		vec4, float, NerfPayload, // m_rays_hit

		network_precision_t,
		float,
		uint32_t,
		uint32_t
	>(
		stream, &m_scratch_alloc,
		n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements,
		n_elements * MAX_STEPS_INBETWEEN_COMPACTION * padded_output_width,
		n_elements * MAX_STEPS_INBETWEEN_COMPACTION * num_floats,
		32, // 2 full cache lines to ensure no overlap
		32  // 2 full cache lines to ensure no overlap
	);
	m_rays[0].set(std::get<0>(scratch), std::get<1>(scratch), std::get<2>(scratch), n_elements);
	m_rays[1].set(std::get<3>(scratch), std::get<4>(scratch), std::get<5>(scratch), n_elements);
	m_rays_hit.set(std::get<6>(scratch), std::get<7>(scratch), std::get<8>(scratch), n_elements);
	m_network_output = std::get<9>(scratch);
	m_network_input = std::get<10>(scratch);

	m_hit_counter = std::get<11>(scratch);
	m_alive_counter = std::get<12>(scratch);
}

// init rays
void NerfJobs::NerfTracer::j01(
	uint32_t sample_index,
	uint32_t padded_output_width,
	uint32_t n_extra_dims,
	const ivec2& resolution,
	const vec2& focal_length,
	const mat4x3& camera_matrix0,
	const mat4x3& camera_matrix1,
	const vec4& rolling_shutter,
	const vec2& screen_center,
	const vec3& parallax_shift,
	bool snap_to_pixel_centers,
	const BoundingBox& render_aabb,
	const mat3& render_aabb_to_local,
	float near_distance,
	float plane_z,
	float aperture_size,
	const Foveation& foveation,
	const Lens& lens,
	const Buffer2DView<const vec4>& envmap,
	const Buffer2DView<const vec2>& distortion,
	vec4* frame_buffer,
	float* depth_buffer,
	const Buffer2DView<const uint8_t>& hidden_area_mask,
	const uint8_t* grid,
	int show_accel,
	uint32_t max_mip,
	float cone_angle_constant,
	int n_pixels,
	int y_offset,
	int y_stride,
	ERenderMode render_mode,
	RaysNerfSoa *m_rays,
	RaysNerfSoa &m_rays_hit,
	cudaStream_t stream
) {
	// Make sure we have enough memory reserved to render at the requested resolution
	// size_t n_pixels = (size_t)resolution.x * resolution.y;
	// enlarge(n_pixels, padded_output_width, n_extra_dims, stream);
	
	enlarge_multi_devices(n_pixels, padded_output_width, n_extra_dims, m_rays,m_rays_hit,stream);
	CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].rgba, 0, 1 * sizeof(vec4), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].depth, 0, 1 * sizeof(float), stream));
	
	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)resolution.x, threads.x), div_round_up((uint32_t)resolution.y, threads.y), 1 };
	
	init_rays_with_payload_kernel_jobs<<<blocks, threads, 0, stream>>>(
		sample_index,
		m_rays[0].payload,
		resolution,
		focal_length,
		camera_matrix0,
		camera_matrix1,
		rolling_shutter,
		screen_center,
		parallax_shift,
		snap_to_pixel_centers,
		render_aabb,
		render_aabb_to_local,
		near_distance,
		plane_z,
		aperture_size,
		foveation,
		lens,
		envmap,
		frame_buffer,
		depth_buffer,
		hidden_area_mask,
		distortion,
		render_mode
	);

	m_n_rays_initialized = resolution.x * y_stride;

	CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].rgba, 0, m_n_rays_initialized * sizeof(vec4), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].depth, 0, m_n_rays_initialized * sizeof(float), stream));

	linear_kernel(advance_pos_nerf_jobs, 0, stream,
		m_n_rays_initialized,
		render_aabb,
		render_aabb_to_local,
		camera_matrix1[2],
		focal_length,
		sample_index,
		m_rays[0].payload,
		grid,
		(show_accel >= 0) ? show_accel : 0,
		max_mip,
		cone_angle_constant
	);
}

__device__ float warp_dt_jobs(float dt) {
	float max_stepsize = MIN_CONE_STEPSIZE() * (1<<(NERF_CASCADES()-1));
	return (dt - MIN_CONE_STEPSIZE()) / (max_stepsize - MIN_CONE_STEPSIZE());
}


__global__ void generate_next_nerf_network_inputs_jobs(
	const uint32_t n_elements,
	BoundingBox render_aabb,
	mat3 render_aabb_to_local,
	BoundingBox train_aabb,
	vec2 focal_length,
	vec3 camera_fwd,
	NerfPayload* __restrict__ payloads,
	PitchedPtr<NerfCoordinate> network_input,
	uint32_t n_steps,
	const uint8_t* __restrict__ density_grid,
	uint32_t min_mip,
	uint32_t max_mip,
	float cone_angle_constant,
	const float* extra_dims
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	NerfPayload& payload = payloads[i];

	if (!payload.alive) {
		return;
	}

	vec3 origin = payload.origin;
	vec3 dir = payload.dir;
	vec3 idir = vec3(1.0f) / dir;

	float cone_angle = cone_angle_constant;

	float t = payload.t;

	for (uint32_t j = 0; j < n_steps; ++j) {
		t = if_unoccupied_advance_to_next_occupied_voxel_jobs(t, cone_angle, {origin, dir}, idir, density_grid, min_mip, max_mip, render_aabb, render_aabb_to_local);
		if (t >= MAX_DEPTH()) {
			payload.n_steps = j;
			return;
		}

		float dt = advance_n_steps_jobs(t, cone_angle, 1.0f) - t;//calc_dt(t, cone_angle);
		network_input(i + j * n_elements)->set_with_optional_extra_dims(train_aabb.relative_pos(origin + dir * t), (dir + vec3(1.0f)) * 0.5f, warp_dt_jobs(dt), extra_dims, network_input.stride_in_bytes); // XXXCONE
		t += dt;
	}

	payload.t = t;
	payload.n_steps = n_steps;
}



// sample rays
int NerfJobs::NerfTracer::j1(NerfNetwork<network_precision_t>& network,
			const BoundingBox& render_aabb,
			const mat3& render_aabb_to_local,
			const BoundingBox& train_aabb,
			const vec2& focal_length,
			float cone_angle_constant,
			const uint8_t* grid,
			ERenderMode render_mode,
			const mat4x3 &camera_matrix,
			float depth_scale,
			int visualized_layer,
			int visualized_dim,
			ENerfActivation rgb_activation,
			ENerfActivation density_activation,
			int show_accel,
			uint32_t max_mip,
			float min_transmittance,
			float glow_y_cutoff,
			int glow_mode,
			const float* extra_dims_gpu,
			RaysNerfSoa *m_rays,
			RaysNerfSoa &m_rays_hit,
			int double_buffer_index,
			int n_alive,
			uint32_t &n_elements,
			uint32_t &extra_stride,
			cudaStream_t stream
			){
    RaysNerfSoa& rays_current = m_rays[(double_buffer_index + 1) % 2];
	RaysNerfSoa& rays_tmp = m_rays[double_buffer_index % 2];
	// ++double_buffer_index;

		// Compact rays that did not diverge yet
		{
			CUDA_CHECK_THROW(cudaMemsetAsync(m_alive_counter, 0, sizeof(uint32_t), stream));
			linear_kernel(compact_kernel_nerf_jobs, 0, stream,
				n_alive,
				rays_tmp.rgba, rays_tmp.depth, rays_tmp.payload,
				rays_current.rgba, rays_current.depth, rays_current.payload,
				m_rays_hit.rgba, m_rays_hit.depth, m_rays_hit.payload,
				m_alive_counter, m_hit_counter
			);
			
			CUDA_CHECK_THROW(cudaMemcpyAsync(&n_alive, m_hit_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
			std::cout << "m_alive_counter: " << n_alive << std::endl;		
			CUDA_CHECK_THROW(cudaMemcpyAsync(&n_alive, m_alive_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
		}
		if(n_alive == 0)
			return 0;
		uint32_t n_steps_between_compaction = tcnn::clamp(m_n_rays_initialized / n_alive, (uint32_t)MIN_STEPS_INBETWEEN_COMPACTION, (uint32_t)MAX_STEPS_INBETWEEN_COMPACTION);
		
		extra_stride = network.n_extra_dims() * sizeof(float);
		PitchedPtr<NerfCoordinate> input_data((NerfCoordinate*)m_network_input, 1, 0, extra_stride);
		linear_kernel(generate_next_nerf_network_inputs_jobs, 0, stream,
			n_alive,
			render_aabb,
			render_aabb_to_local,
			train_aabb,
			focal_length,
			camera_matrix[2],
			rays_current.payload,
			input_data,
			n_steps_between_compaction,
			grid,
			(show_accel>=0) ? show_accel : 0,
			max_mip,
			cone_angle_constant,
			extra_dims_gpu
		);
		std::cout << "n_alive: " << n_alive << std::endl;
		n_elements = next_multiple(n_alive * n_steps_between_compaction, tcnn::batch_size_granularity);
		return 	n_alive;
}

// encoding
template <typename T>
int NerfJobs::NerfTracer::j20(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>& density_network_input,NerfNetwork<precision_t>& network){
	uint32_t batch_size = input.n();
	network.m_pos_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(0, network.m_pos_encoding->input_width()),
			density_network_input,
			true
		);
	return 0;
}

// encoding
template <typename T>
int NerfJobs::NerfTracer::j21(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>& rgb_network_input,NerfNetwork<precision_t>& network){
	uint32_t batch_size = input.n();
	auto dir_out = rgb_network_input.slice_rows(network.m_density_network->padded_output_width(), network.m_dir_encoding->padded_output_width());
	network.m_dir_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(network.m_dir_offset,network.m_dir_encoding->input_width()),
			dir_out,
			true
		);
	return 0;
}

// mlp
template <typename T>
int NerfJobs::NerfTracer::j30(cudaStream_t stream, const tcnn::GPUMatrixDynamic<T>& density_network_input, tcnn::GPUMatrixDynamic<T>& density_network_output,NerfNetwork<precision_t>& network,uint32_t batch_size){
	network.m_density_network->inference_mixed_precision(stream, density_network_input, density_network_output, true);
	return 0;
}

// mlp
template <typename T>
int NerfJobs::NerfTracer::j31(cudaStream_t stream, const tcnn::GPUMatrixDynamic<T>& rgb_network_input, tcnn::GPUMatrixDynamic<T>& rgb_network_output,
							 tcnn::GPUMatrixDynamic<T>& density_network_output, tcnn::GPUMatrixDynamic<T>& output,NerfNetwork<precision_t>& network,uint32_t batch_size){
	network.m_rgb_network->inference_mixed_precision(stream, rgb_network_input, rgb_network_output, true);

	tcnn::linear_kernel(extract_density<T>, 0, stream,
		batch_size,
		density_network_output.layout() == tcnn::AoS ? density_network_output.stride() : 1,
		output.layout() == tcnn::AoS ? network.padded_output_width() : 1,
		density_network_output.data(),
		output.data() + 3 * (output.layout() == tcnn::AoS ? 1 : batch_size)
	);
	return 0;
}

// get rgb
int j4(){

}

vec2 NerfJobs::calc_focal_length(const ivec2& resolution, const vec2& relative_focal_length, int fov_axis, float zoom) const {
	return relative_focal_length * (float)resolution[fov_axis] * zoom;
}

vec2 NerfJobs::render_screen_center(const vec2& screen_center) const {
	// see pixel_to_ray for how screen center is used; 0.5, 0.5 is 'normal'. we flip so that it becomes the point in the original image we want to center on.
	return (vec2(0.5f) - screen_center) * m_zoom + vec2(0.5f);
}

std::pair<vec2,vec2> NerfJobs::NerfTracer::j00(const ivec2& resolution,const vec2& relative_focal_length,int m_fov_axis,float m_zoom,const vec2& orig_screen_center){
	vec2 focal_length = relative_focal_length * (float)resolution[m_fov_axis] * m_zoom;
	vec2 screen_center = (vec2(0.5f) - orig_screen_center) * m_zoom + vec2(0.5f);
	return std::make_pair(focal_length, screen_center);
}

vec3 NerfJobs::look_at() const {
	return view_pos() + view_dir() * m_scale;
}

void NerfJobs::set_look_at(const vec3& pos) {
	m_camera[3] += pos - look_at();
}

void NerfJobs::set_scale(float scale) {
	auto prev_look_at = look_at();
	m_camera[3] = (view_pos() - prev_look_at) * (scale / m_scale) + prev_look_at;
	m_scale = scale;
}

void NerfJobs::set_view_dir(const vec3& dir) {
	auto old_look_at = look_at();
	m_camera[0] = normalize(cross(dir, m_up_dir));
	m_camera[1] = normalize(cross(dir, m_camera[0]));
	m_camera[2] = normalize(dir);
	set_look_at(old_look_at);
}

void NerfJobs::first_training_view() {
	m_nerf.training.view = 0;
	set_camera_to_training_view(m_nerf.training.view);
}

void NerfJobs::last_training_view() {
	m_nerf.training.view = m_nerf.training.dataset.n_images-1;
	set_camera_to_training_view(m_nerf.training.view);
}

void NerfJobs::previous_training_view() {
	if (m_nerf.training.view != 0) {
		m_nerf.training.view -= 1;
	}

	set_camera_to_training_view(m_nerf.training.view);
}

void NerfJobs::next_training_view() {
	if (m_nerf.training.view != m_nerf.training.dataset.n_images-1) {
		m_nerf.training.view += 1;
	}

	set_camera_to_training_view(m_nerf.training.view);
}


void NerfJobs::NerfTracer::init_rays_from_camera_multi_devices(
	uint32_t sample_index,
	uint32_t padded_output_width,
	uint32_t n_extra_dims,
	const ivec2& resolution,
	const vec2& focal_length,
	const mat4x3& camera_matrix0,
	const mat4x3& camera_matrix1,
	const vec4& rolling_shutter,
	const vec2& screen_center,
	const vec3& parallax_shift,
	bool snap_to_pixel_centers,
	const BoundingBox& render_aabb,
	const mat3& render_aabb_to_local,
	float near_distance,
	float plane_z,
	float aperture_size,
	const Foveation& foveation,
	const Lens& lens,
	const Buffer2DView<const vec4>& envmap,
	const Buffer2DView<const vec2>& distortion,
	vec4* frame_buffer,
	float* depth_buffer,
	const Buffer2DView<const uint8_t>& hidden_area_mask,
	const uint8_t* grid,
	int show_accel,
	uint32_t max_mip,
	float cone_angle_constant,
	int n_pixels,
	int y_offset,
	int y_stride,
	ERenderMode render_mode,
	RaysNerfSoa *m_rays,
	RaysNerfSoa &m_rays_hit,
	cudaStream_t stream
) {
	// Make sure we have enough memory reserved to render at the requested resolution
	enlarge_multi_devices(n_pixels, padded_output_width, n_extra_dims, m_rays,m_rays_hit,stream);
	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)resolution.x, threads.x), div_round_up((uint32_t)y_stride, threads.y), 1 };

	
	init_rays_with_payload_kernel_jobs<<<blocks, threads, 0, stream>>>(
		sample_index,
		m_rays[0].payload,
		resolution,
		focal_length,
		camera_matrix0,
		camera_matrix1,
		rolling_shutter,
		screen_center,
		parallax_shift,
		snap_to_pixel_centers,
		render_aabb,
		render_aabb_to_local,
		near_distance,
		plane_z,
		aperture_size,
		foveation,
		lens,
		envmap,
		frame_buffer,
		depth_buffer,
		hidden_area_mask,
		distortion,
		render_mode
	);
	m_n_rays_initialized = resolution.x * y_stride;

	CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].rgba, 0, m_n_rays_initialized * sizeof(vec4), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].depth, 0, m_n_rays_initialized * sizeof(float), stream));
	linear_kernel(advance_pos_nerf_jobs, 0, stream,
		m_n_rays_initialized,
		render_aabb,
		render_aabb_to_local,
		camera_matrix1[2],
		focal_length,
		sample_index,
		m_rays[0].payload,
		grid,
		(show_accel >= 0) ? show_accel : 0,
		max_mip,
		cone_angle_constant
	);
}

uint32_t NerfJobs::NerfTracer::trace_multi_devices(
	NerfNetwork<network_precision_t>& network,
	const BoundingBox& render_aabb,
	const mat3& render_aabb_to_local,
	const BoundingBox& train_aabb,
	const vec2& focal_length,
	float cone_angle_constant,
	const uint8_t* grid,
	ERenderMode render_mode,
	const mat4x3 &camera_matrix,
	float depth_scale,
	int visualized_layer,
	int visualized_dim,
	ENerfActivation rgb_activation,
	ENerfActivation density_activation,
	int show_accel,
	uint32_t max_mip,
	float min_transmittance,
	float glow_y_cutoff,
	int glow_mode,
	const float* extra_dims_gpu,
	RaysNerfSoa *m_rays,
	RaysNerfSoa &m_rays_hit,
	cudaStream_t stream
) {
	if (m_n_rays_initialized == 0) {
		return 0;
	}
	CUDA_CHECK_THROW(cudaMemsetAsync(m_hit_counter, 0, sizeof(uint32_t), stream));

	uint32_t n_alive = m_n_rays_initialized;
	// m_n_rays_initialized = 0;

	uint32_t i = 1;
	uint32_t double_buffer_index = 0;

	while (i < MARCH_ITER) {
		// RaysNerfSoa& rays_current = m_rays[(double_buffer_index + 1) % 2];
		// RaysNerfSoa& rays_tmp = m_rays[double_buffer_index % 2];
		
		// uint32_t extra_stride = network.n_extra_dims() * sizeof(float);
		// uint32_t n_elements ;//= next_multiple(n_alive * n_steps_between_compaction, tcnn::batch_size_granularity);
		// n_alive = j1(network,
		// 			render_aabb,
		// 			render_aabb_to_local,
		// 			train_aabb,
		// 			focal_length,
		// 			cone_angle_constant,
		// 			grid,
		// 			render_mode,
		// 			camera_matrix,
		// 			depth_scale,
		// 			visualized_layer,
		// 			visualized_dim,
		// 			rgb_activation,
		// 			density_activation,
		// 			show_accel,
		// 			max_mip,
		// 			min_transmittance,
		// 			glow_y_cutoff,
		// 			glow_mode,
		// 			extra_dims_gpu,
		// 			m_rays,
		// 			m_rays_hit,
		// 			double_buffer_index,
		// 			n_alive,
		// 			n_elements,
		// 			extra_stride,
		// 			stream);
		// ++double_buffer_index;
		// if (n_alive == 0) {
		// 	break;
		// }
		// // Want a large number of queries to saturate the GPU and to ensure compaction doesn't happen toooo frequently.
		// uint32_t target_n_queries = 2 * 1024 * 1024;
		// uint32_t n_steps_between_compaction = tcnn::clamp(target_n_queries / n_alive, (uint32_t)MIN_STEPS_INBETWEEN_COMPACTION, (uint32_t)MAX_STEPS_INBETWEEN_COMPACTION);
		// PitchedPtr<NerfCoordinate> input_data((NerfCoordinate*)m_network_input, 1, 0, extra_stride);
		// // linear_kernel(generate_next_nerf_network_inputs_jobs, 0, stream,
		// // 	n_alive,
		// // 	render_aabb,
		// // 	render_aabb_to_local,
		// // 	train_aabb,
		// // 	focal_length,
		// // 	camera_matrix[2],
		// // 	rays_current.payload,
		// // 	input_data,
		// // 	n_steps_between_compaction,
		// // 	grid,
		// // 	(show_accel>=0) ? show_accel : 0,
		// // 	max_mip,
		// // 	cone_angle_constant,
		// // 	extra_dims_gpu
		// // );
		
		RaysNerfSoa& rays_current = m_rays[(double_buffer_index + 1) % 2];
		RaysNerfSoa& rays_tmp = m_rays[double_buffer_index % 2];
		++double_buffer_index;
		// Compact rays that did not diverge yet
		{	
			
			CUDA_CHECK_THROW(cudaMemsetAsync(m_alive_counter, 0, sizeof(uint32_t), stream));
			NerfPayload& src_payload = rays_tmp.payload[i];
			linear_kernel(compact_kernel_nerf_jobs, 0, stream,
				n_alive,
				rays_tmp.rgba, rays_tmp.depth, rays_tmp.payload,
				rays_current.rgba, rays_current.depth, rays_current.payload,
				m_rays_hit.rgba, m_rays_hit.depth, m_rays_hit.payload,
				m_alive_counter, m_hit_counter
			);
			
			CUDA_CHECK_THROW(cudaMemcpyAsync(&n_alive, m_alive_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
		}
		
		if (n_alive == 0) {
			break;
		}
		// Want a large number of queries to saturate the GPU and to ensure compaction doesn't happen toooo frequently.
		uint32_t target_n_queries = 2 * 1024 * 1024;
		uint32_t n_steps_between_compaction = tcnn::clamp(target_n_queries / n_alive, (uint32_t)MIN_STEPS_INBETWEEN_COMPACTION, (uint32_t)MAX_STEPS_INBETWEEN_COMPACTION);
		uint32_t extra_stride = network.n_extra_dims() * sizeof(float);
		PitchedPtr<NerfCoordinate> input_data((NerfCoordinate*)m_network_input, 1, 0, extra_stride);
		linear_kernel(generate_next_nerf_network_inputs_jobs, 0, stream,
			n_alive,
			render_aabb,
			render_aabb_to_local,
			train_aabb,
			focal_length,
			camera_matrix[2],
			rays_current.payload,
			input_data,
			n_steps_between_compaction,
			grid,
			(show_accel>=0) ? show_accel : 0,
			max_mip,
			cone_angle_constant,
			extra_dims_gpu
		);
		
		uint32_t n_elements = next_multiple(n_alive * n_steps_between_compaction, tcnn::batch_size_granularity);
		GPUMatrix<float> positions_matrix((float*)m_network_input, (sizeof(NerfCoordinate) + extra_stride) / sizeof(float), n_elements);
		GPUMatrix<network_precision_t, RM> rgbsigma_matrix((network_precision_t*)m_network_output, network.padded_output_width(), n_elements);


		// network.inference_mixed_precision(stream, positions_matrix, rgbsigma_matrix);

		uint32_t batch_size = positions_matrix.n();
		tcnn::GPUMatrixDynamic<network_precision_t> density_network_input{network.m_pos_encoding->padded_output_width(), batch_size, stream, network.m_pos_encoding->preferred_output_layout()};
		tcnn::GPUMatrixDynamic<network_precision_t> rgb_network_input{network.m_rgb_network_input_width, batch_size, stream, network.m_dir_encoding->preferred_output_layout()};

		tcnn::GPUMatrixDynamic<network_precision_t> density_network_output = rgb_network_input.slice_rows(0, network.m_density_network->padded_output_width());
		tcnn::GPUMatrixDynamic<network_precision_t> rgb_network_output{rgbsigma_matrix.data(), network.m_rgb_network->padded_output_width(), batch_size, rgbsigma_matrix.layout()};
		
		j20(stream, positions_matrix, density_network_input,network);
		j30(stream,density_network_input,density_network_output,network,batch_size);
		j21(stream, positions_matrix, rgb_network_input,network);
		j31(stream, rgb_network_input,rgb_network_output,density_network_output, rgbsigma_matrix,network,batch_size);

		if (render_mode == ERenderMode::Normals) {
			network.input_gradient(stream, 3, positions_matrix, positions_matrix);
		} else if (render_mode == ERenderMode::EncodingVis) {
			network.visualize_activation(stream, visualized_layer, visualized_dim, positions_matrix, positions_matrix);
		}

		
		linear_kernel(composite_kernel_nerf_jobs, 0, stream,
			n_alive,
			n_elements,
			i,
			train_aabb,
			glow_y_cutoff,
			glow_mode,
			camera_matrix,
			focal_length,
			depth_scale,
			rays_current.rgba,
			rays_current.depth,
			rays_current.payload,
			input_data,
			m_network_output,
			network.padded_output_width(),
			n_steps_between_compaction,
			render_mode,
			grid,
			rgb_activation,
			density_activation,
			show_accel,
			min_transmittance
		);
		i += n_steps_between_compaction;
	}
	uint32_t n_hit;
	CUDA_CHECK_THROW(cudaMemcpyAsync(&n_hit, m_hit_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
	// for(int i = 0; i < e_start_layers.size(); ++i){
	// 	cudaEventSynchronize(e_stop_layers[i]);
	// 	float elapsedTime;
	// 	cudaEventElapsedTime(&elapsedTime, e_start_layers[i], e_stop_layers[i]);
	// 	total_time += elapsedTime;
		
	// }
	// printf("Time %d to : %3.3f ms.\n", i,total_time);
	return n_hit;
}

void NerfJobs::render_nerf_multi_device(
	std::vector<CudaDevice>& devices,
    const vec2& focal_length,
    const mat4x3& camera_matrix0,
    const mat4x3& camera_matrix1,
    const vec4& rolling_shutter,
    const vec2& screen_center,
    const Foveation& foveation,
    int visualized_dimension
) {

	auto render_buffer = devices[0].render_buffer_view();
    int n_devices = static_cast<int>(devices.size());
    int total_num_rays = render_buffer.resolution.x * render_buffer.resolution.y;
	int parallelism_degree = 1;
    // int y_stride = (render_buffer.resolution.y + n_devices - 1) / n_devices; // Round up to ensure all rays are processed
	int y_stride = (render_buffer.resolution.y + n_devices - 1) / parallelism_degree;
    // Loop over the devices and render the rays assigned to each device
	//use openmp to parallel
	omp_set_nested(parallelism_degree);
	#pragma omp parallel for num_threads(parallelism_degree)
    for (int i = 0; i < parallelism_degree; ++i) {
		// cudaSetDevice(i);
		
        cudaStream_t stream = devices[i].stream();
		// cudaStreamCreate(&stream);
        const CudaRenderBufferView& render_buffer = devices[i].render_buffer_view();
		auto density_grid_bitfield = devices[i].data().density_grid_bitfield_ptr;
		int y_offset = i * y_stride;
		int num_rays = y_stride * render_buffer.resolution.x;
		RaysNerfSoa m_rays[2];
		RaysNerfSoa m_rays_hit;
		// std::cout << camera_matrix1[0][0] << " " << camera_matrix1[0][1] << " " << camera_matrix1[0][2] << std::endl;
		// std::cout << camera_matrix1[1][0] << " " << camera_matrix1[1][1] << " " << camera_matrix1[1][2] << std::endl;
		// std::cout << camera_matrix1[2][0] << " " << camera_matrix1[2][1] << " " << camera_matrix1[2][2] << std::endl;

		auto nerf_network = *devices[i].nerf_network();
        // Initialize the rays for this device
        NerfTracer tracer;
        tracer.j01(
            render_buffer.spp,
            nerf_network.padded_output_width(),
            nerf_network.n_extra_dims(),
            render_buffer.resolution,
            focal_length,
            camera_matrix0,
            camera_matrix1,
            rolling_shutter,
            screen_center,
            m_parallax_shift,
            m_snap_to_pixel_centers,
            m_render_aabb,
            m_render_aabb_to_local,
            m_render_near_distance,
            m_slice_plane_z + m_scale,
            m_aperture_size,
            foveation,
            m_nerf.render_with_lens_distortion ? m_nerf.render_lens : Lens{},
            m_envmap.inference_view(),
            m_nerf.render_with_lens_distortion && m_dlss ? m_distortion.inference_view() : Buffer2DView<const vec2>{},
            render_buffer.frame_buffer,
            render_buffer.depth_buffer,
            render_buffer.hidden_area_mask ? render_buffer.hidden_area_mask->const_view() : Buffer2DView<const uint8_t>{},
            density_grid_bitfield,
            m_nerf.show_accel,
            m_nerf.max_cascade,
            m_nerf.cone_angle_constant,
			num_rays,
			y_offset,
			y_stride,
            m_render_mode,
			m_rays,
			m_rays_hit,
            stream
        );

		
        // Trace the rays for this device
        uint32_t n_hits = 0;
        if (m_render_mode != ERenderMode::Slice) {
            float depth_scale = 1.0f / m_nerf.training.dataset.scale;
            n_hits = tracer.trace_multi_devices(
                nerf_network,
                m_render_aabb,
                m_render_aabb_to_local,
                m_aabb,
                focal_length,
                m_nerf.cone_angle_constant,
                density_grid_bitfield,
                m_render_mode,
                camera_matrix1,
                depth_scale,
                m_visualized_layer,
                visualized_dimension,
                m_nerf.rgb_activation,
                m_nerf.density_activation,
                m_nerf.show_accel,
                m_nerf.max_cascade,
                m_nerf.render_min_transmittance,
                m_nerf.glow_y_cutoff,
                m_nerf.glow_mode,
                get_inference_extra_dims(stream),
				m_rays,
				m_rays_hit,
				stream
            );
        }
		RaysNerfSoa& rays_hit = m_render_mode == ERenderMode::Slice ? tracer.rays_init() :m_rays_hit;
		linear_kernel(shade_kernel_nerf_jobs, 0, stream,
			n_hits,
			rays_hit.rgba,
			rays_hit.depth,
			rays_hit.payload,
			m_render_mode,
			m_nerf.training.linear_colors,
			render_buffer.frame_buffer + i * render_buffer.resolution.x *y_stride,
			render_buffer.depth_buffer + i * render_buffer.resolution.x *y_stride
		);	
		if (i > 1)	{
			uint32_t offside = i * y_stride * render_buffer.resolution.x;
			cudaMemcpyAsync(devices[0].render_buffer_view().frame_buffer + offside, devices[i].render_buffer_view().frame_buffer, y_stride * render_buffer.resolution.x * sizeof(vec4) , cudaMemcpyDeviceToDevice, devices[i].stream());
			cudaMemcpyAsync(devices[0].render_buffer_view().depth_buffer + offside, devices[i].render_buffer_view().depth_buffer, y_stride * render_buffer.resolution.x * sizeof(float), cudaMemcpyDeviceToDevice, devices[i].stream());
		}
		// cudaStreamDestroy(stream);
	}
	// for (int i = 0; i < n_devices; ++i) {
	// 	std::thread t(render_nerf_multi_device_thread1);
	// 	t.detach();
	// }
	#pragma omp barrier
	
	// copy every render_buffer to the first device
	for (int i = 0; i < n_devices; ++i) {
		cudaStreamSynchronize(devices[i].stream());
	}
    // Synchronize all streams before returning
}


const float* NerfJobs::get_inference_extra_dims(cudaStream_t stream) const {
	if (m_nerf_network->n_extra_dims() == 0) {
		return nullptr;
	}
	const float* extra_dims_src = m_nerf.training.extra_dims_gpu.data() + m_nerf.extra_dim_idx_for_inference * m_nerf.training.dataset.n_extra_dims();
	if (!m_nerf.training.dataset.has_light_dirs) {
		return extra_dims_src;
	}

	// the dataset has light directions, so we must construct a temporary buffer and fill it as requested.
	// we use an extra 'slot' that was pre-allocated for us at the end of the extra_dims array.
	size_t size = m_nerf_network->n_extra_dims() * sizeof(float);
	float* dims_gpu = m_nerf.training.extra_dims_gpu.data() + m_nerf.training.dataset.n_images * m_nerf.training.dataset.n_extra_dims();
	CUDA_CHECK_THROW(cudaMemcpyAsync(dims_gpu, extra_dims_src, size, cudaMemcpyDeviceToDevice, stream));
	vec3 light_dir = warp_direction_jobs(normalize(m_nerf.light_dir));
	CUDA_CHECK_THROW(cudaMemcpyAsync(dims_gpu, &light_dir, min(size, sizeof(vec3)), cudaMemcpyHostToDevice, stream));
	return dims_gpu;
}









void NerfJobs::render_frame_main_multi_devices(
	std::vector<CudaDevice>& devices,
	const mat4x3& camera_matrix0,
	const mat4x3& camera_matrix1,
	const vec2& orig_screen_center,
	const vec2& relative_focal_length,
	const vec4& nerf_rolling_shutter,
	const Foveation& foveation,
	int visualized_dimension
){
	
	// for (const auto& device : devices) {
	// 	device.render_buffer_view().clear(device.stream());
	// }

	// if (!m_network) {
	// 	return;
	// }
	// NerfTracer tracer;
	
	// auto focal_and_center = tracer.j00(devices[0].render_buffer_view().resolution,relative_focal_length,m_fov_axis,m_zoom,orig_screen_center);
	// vec2 focal_length = focal_and_center.first;
	// vec2 screen_center = focal_and_center.second;
	
	// std::cout <<  m_ground_truth_alpha <<std::endl;

	// if (m_render_ground_truth && m_ground_truth_alpha >= 1.0f){
	// 	return;
	// }
	// auto render_buffer = devices[0].render_buffer_view();
    // int n_devices = static_cast<int>(devices.size());
    // int total_num_rays = render_buffer.resolution.x * render_buffer.resolution.y;
    // int y_stride = (render_buffer.resolution.y + n_devices - 1) / n_devices; // Round up to ensure all rays are processed
	
	// for (int i = 0; i < 1; ++i) {
	// 	cudaSetDevice(i);
    //     cudaStream_t stream = devices[i].stream();
    //     const CudaRenderBufferView& render_buffer = devices[i].render_buffer_view();
	// 	auto density_grid_bitfield = devices[i].data().density_grid_bitfield_ptr;
	// 	int y_offset = i * y_stride;
	// 	int num_rays = y_stride * render_buffer.resolution.x;
	// 	RaysNerfSoa m_rays[2];
	// 	RaysNerfSoa m_rays_hit;
		
	// 	auto nerf_network = *devices[i].nerf_network();
	// 	// std::cout << camera_matrix1[0][0] << " " << camera_matrix1[0][1] << " " << camera_matrix1[0][2] << std::endl;
	// 	// std::cout << camera_matrix1[1][0] << " " << camera_matrix1[1][1] << " " << camera_matrix1[1][2] << std::endl;
	// 	// std::cout << camera_matrix1[2][0] << " " << camera_matrix1[2][1] << " " << camera_matrix1[2][2] << std::endl;
	// 	std::cout << screen_center[0] << " " << screen_center[1] << " " << nerf_rolling_shutter[2] << " " << nerf_rolling_shutter[3] <<std::endl;
    //     // Initialize the rays for this device
	// 	tracer.j01(render_buffer.spp,
    //         nerf_network.padded_output_width(),
    //         nerf_network.n_extra_dims(),
    //         render_buffer.resolution,
    //         focal_length,
    //         camera_matrix0,
    //         camera_matrix1,
    //         nerf_rolling_shutter,
    //         screen_center,
    //         m_parallax_shift,
    //         m_snap_to_pixel_centers,
    //         m_render_aabb,
    //         m_render_aabb_to_local,
    //         m_render_near_distance,
    //         m_slice_plane_z + m_scale,
    //         m_aperture_size,
    //         foveation,
    //         m_nerf.render_with_lens_distortion ? m_nerf.render_lens : Lens{},
    //         m_envmap.inference_view(),
    //         m_nerf.render_with_lens_distortion && m_dlss ? m_distortion.inference_view() : Buffer2DView<const vec2>{},
    //         render_buffer.frame_buffer,
    //         render_buffer.depth_buffer,
    //         render_buffer.hidden_area_mask ? render_buffer.hidden_area_mask->const_view() : Buffer2DView<const uint8_t>{},
    //         density_grid_bitfield,
    //         m_nerf.show_accel,
    //         m_nerf.max_cascade,
    //         m_nerf.cone_angle_constant,
	// 		num_rays,
	// 		y_offset,
	// 		y_stride,
    //         m_render_mode,
	// 		m_rays,
	// 		m_rays_hit,
    //         stream);
	// 	uint32_t n_hits = 0;
	// 	if (m_render_mode != ERenderMode::Slice) {
	// 		float depth_scale = 1.0f / m_nerf.training.dataset.scale;
	// 		CUDA_CHECK_THROW(cudaMemsetAsync(tracer.m_hit_counter, 0, sizeof(uint32_t), stream));
	// 		uint32_t i = 1;
	// 		uint32_t double_buffer_index = 0;
	// 		uint32_t n_alive = tracer.n_rays_initialized();

	// 		while (i < MARCH_ITER) {

	// 			uint32_t n_elements;
	// 			uint32_t extra_stride;
	// 			RaysNerfSoa& rays_current = m_rays[(double_buffer_index + 1) % 2];
	// 			RaysNerfSoa& rays_tmp = m_rays[double_buffer_index % 2];
	// 			std::cout << 123456 << " " << n_alive<<std::endl;
	// 			n_alive = tracer.j1(nerf_network,
	// 				m_render_aabb,
	// 				m_render_aabb_to_local,
	// 				m_aabb,
	// 				focal_length,
	// 				m_nerf.cone_angle_constant,
	// 				density_grid_bitfield,
	// 				m_render_mode,
	// 				camera_matrix1,
	// 				depth_scale,
	// 				m_visualized_layer,
	// 				visualized_dimension,
	// 				m_nerf.rgb_activation,
	// 				m_nerf.density_activation,
	// 				m_nerf.show_accel,
	// 				m_nerf.max_cascade,
	// 				m_nerf.render_min_transmittance,
	// 				m_nerf.glow_y_cutoff,
	// 				m_nerf.glow_mode,
	// 				get_inference_extra_dims(stream),
	// 				m_rays,
	// 				m_rays_hit,
	// 				double_buffer_index,
	// 				n_alive,
	// 				n_elements,
	// 				extra_stride,
	// 				stream);
	// 			uint32_t target_n_queries = 2 * 1024 * 1024;
	// 			uint32_t n_steps_between_compaction = tcnn::clamp(target_n_queries / n_alive, (uint32_t)MIN_STEPS_INBETWEEN_COMPACTION, (uint32_t)MAX_STEPS_INBETWEEN_COMPACTION);
	// 			PitchedPtr<NerfCoordinate> input_data((NerfCoordinate*)tracer.m_network_input, 1, 0, extra_stride);
	// 			GPUMatrix<float> positions_matrix((float*)tracer.m_network_input, (sizeof(NerfCoordinate) + extra_stride) / sizeof(float), n_elements);
	// 			GPUMatrix<network_precision_t, RM> rgbsigma_matrix((network_precision_t*)tracer.m_network_output, nerf_network.padded_output_width(), n_elements);
	// 			// uint32_t batch_size = positions_matrix.n();
	// 			// tcnn::GPUMatrixDynamic<network_precision_t> density_network_input{nerf_network.m_pos_encoding->padded_output_width(), batch_size, stream, nerf_network.m_pos_encoding->preferred_output_layout()};
	// 			// tcnn::GPUMatrixDynamic<network_precision_t> rgb_network_input{nerf_network.m_rgb_network_input_width, batch_size, stream, nerf_network.m_dir_encoding->preferred_output_layout()};

	// 			// tcnn::GPUMatrixDynamic<network_precision_t> density_network_output = rgb_network_input.slice_rows(0, nerf_network.m_density_network->padded_output_width());
	// 			// tcnn::GPUMatrixDynamic<network_precision_t> rgb_network_output{rgbsigma_matrix.data(), nerf_network.m_rgb_network->padded_output_width(), batch_size, rgbsigma_matrix.layout()};

	// 			// tracer.j20(stream, positions_matrix, density_network_input,nerf_network);
	// 			// tracer.j30(stream,density_network_input,density_network_output,nerf_network,batch_size);
	// 			// tracer.j21(stream, positions_matrix, rgb_network_input,nerf_network);
	// 			// tracer.j31(stream, rgb_network_input,rgb_network_output,density_network_output, rgbsigma_matrix,nerf_network,batch_size);
	// 			nerf_network.inference_mixed_precision(stream, positions_matrix, rgbsigma_matrix);
				
	// 			linear_kernel(composite_kernel_nerf_jobs, 0, stream,
	// 				n_alive,
	// 				n_elements,
	// 				i,
	// 				m_aabb,
	// 				m_nerf.glow_y_cutoff,
	// 				m_nerf.glow_mode,
	// 				camera_matrix1,
	// 				focal_length,
	// 				depth_scale,
	// 				rays_current.rgba,
	// 				rays_current.depth,
	// 				rays_current.payload,
	// 				input_data,
	// 				tracer.m_network_output,
	// 				nerf_network.padded_output_width(),
	// 				n_steps_between_compaction,
	// 				m_render_mode,
	// 				density_grid_bitfield,
	// 				m_nerf.rgb_activation,
	// 				m_nerf.density_activation,
	// 				m_nerf.show_accel,
	// 				m_nerf.render_min_transmittance
	// 			);
	// 			std::cout << 12345 << std::endl;
	// 			if(i==3)
	// 				exit(0);
	// 			i += n_steps_between_compaction;
	// 			std::cout << n_steps_between_compaction << std::endl;
	// 		}

	// 		uint32_t n_hit;
	// 		CUDA_CHECK_THROW(cudaMemcpyAsync(&n_hit, tracer.m_hit_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
	// 		CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

	// 		RaysNerfSoa& rays_hit = m_render_mode == ERenderMode::Slice ? tracer.rays_init() :m_rays_hit;
	// 		linear_kernel(shade_kernel_nerf_jobs, 0, stream,
	// 			n_hits,
	// 			rays_hit.rgba,
	// 			rays_hit.depth,
	// 			rays_hit.payload,
	// 			m_render_mode,
	// 			m_nerf.training.linear_colors,
	// 			render_buffer.frame_buffer,
	// 			render_buffer.depth_buffer
	// 		);	
	// 	}
		
	// }


	for (const auto& device : devices) {
		device.render_buffer_view().clear(device.stream());
	}

	if (!m_network) {
		return;
	}
	// 计算焦距和屏幕中心位置
	NerfTracer tracer;
	auto focal_and_center = tracer.j00(devices[0].render_buffer_view().resolution,relative_focal_length,m_fov_axis,m_zoom,orig_screen_center);
	vec2 focal_length = focal_and_center.first;
	vec2 screen_center = focal_and_center.second;
	
	switch (m_testbed_mode) {
		// 计算焦距和屏幕中心位置
		case ETestbedMode::Nerf:
			if (!m_render_ground_truth || m_ground_truth_alpha < 1.0f) {
				render_nerf_multi_device(devices, focal_length, camera_matrix0, camera_matrix1, nerf_rolling_shutter, screen_center, foveation, visualized_dimension);
			}
			break;
		default:
			// No-op if no mode is active
			break;
	}
}

void NerfJobs::j4(
	cudaStream_t stream,
	const mat4x3& camera_matrix0,
	const mat4x3& prev_camera_matrix,
	const vec2& orig_screen_center,
	const vec2& relative_focal_length,
	const Foveation& foveation,
	const Foveation& prev_foveation,
	CudaRenderBuffer& render_buffer,
	bool to_srgb
) {
	vec2 focal_length = calc_focal_length(render_buffer.in_resolution(), relative_focal_length, m_fov_axis, m_zoom);
	vec2 screen_center = render_screen_center(orig_screen_center);

	render_buffer.set_color_space(m_color_space);
	render_buffer.set_tonemap_curve(m_tonemap_curve);

	Lens lens = (m_testbed_mode == ETestbedMode::Nerf && m_nerf.render_with_lens_distortion) ? m_nerf.render_lens : Lens{};
	

	EColorSpace output_color_space = to_srgb ? EColorSpace::SRGB : EColorSpace::Linear;


	render_buffer.accumulate(m_exposure, stream);
	render_buffer.tonemap(m_exposure, m_background_color, output_color_space, m_ndc_znear, m_ndc_zfar, m_snap_to_pixel_centers, stream);

	if (m_testbed_mode == ETestbedMode::Nerf) {
		// Overlay the ground truth image if requested
		if (m_render_ground_truth) {
			auto const& metadata = m_nerf.training.dataset.metadata[m_nerf.training.view];
			if (m_ground_truth_render_mode == EGroundTruthRenderMode::Shade) {
				render_buffer.overlay_image(
					m_ground_truth_alpha,
					vec3(m_exposure) + m_nerf.training.cam_exposure[m_nerf.training.view].variable(),
					m_background_color,
					output_color_space,
					metadata.pixels,
					metadata.image_data_type,
					metadata.resolution,
					m_fov_axis,
					m_zoom,
					vec2(0.5f),
					stream
				);
			} else if (m_ground_truth_render_mode == EGroundTruthRenderMode::Depth && metadata.depth) {
				render_buffer.overlay_depth(
					m_ground_truth_alpha,
					metadata.depth,
					1.0f/m_nerf.training.dataset.scale,
					metadata.resolution,
					m_fov_axis,
					m_zoom,
					vec2(0.5f),
					stream
				);
			}
		}

		// Visualize the accumulated error map if requested
		if (m_nerf.training.render_error_overlay) {
			const float* err_data = m_nerf.training.error_map.data.data();
			ivec2 error_map_res = m_nerf.training.error_map.resolution;
			if (m_render_ground_truth) {
				err_data = m_nerf.training.dataset.sharpness_data.data();
				error_map_res = m_nerf.training.dataset.sharpness_resolution;
			}
			size_t emap_size = error_map_res.x * error_map_res.y;
			err_data += emap_size * m_nerf.training.view;

			GPUMemory<float> average_error;
			average_error.enlarge(1);
			average_error.memset(0);
			const float* aligned_err_data_s = (const float*)(((size_t)err_data)&~15);
			const float* aligned_err_data_e = (const float*)(((size_t)(err_data+emap_size))&~15);
			size_t reduce_size = aligned_err_data_e - aligned_err_data_s;
			reduce_sum(aligned_err_data_s, [reduce_size] __device__ (float val) { return max(val,0.f) / (reduce_size); }, average_error.data(), reduce_size, stream);
			auto const &metadata = m_nerf.training.dataset.metadata[m_nerf.training.view];
			render_buffer.overlay_false_color(metadata.resolution, to_srgb, m_fov_axis, stream, err_data, error_map_res, average_error.data(), m_nerf.training.error_overlay_brightness, m_render_ground_truth);
		}
	}
}

void NerfJobs::render_frame_multi_devices(
	const mat4x3& camera_matrix0,
	const mat4x3& camera_matrix1,
	const mat4x3& prev_camera_matrix,
	const vec2& orig_screen_center,
	const vec2& relative_focal_length,
	const vec4& nerf_rolling_shutter,
	const Foveation& foveation,
	const Foveation& prev_foveation,
	int visualized_dimension,
	std::vector<CudaRenderBuffer>& render_buffer,
	bool to_srgb,
	CudaDevice* device
) {
	for(int i = 0;i< m_devices.size();i++){
		
		sync_device(render_buffer[i], m_devices[i]);
	}
	
	std::vector<ScopeGuard> guards;
	{
		for (int i = 0;i< m_devices.size();i++){

			auto device_guard = use_device(m_devices[i].stream(), render_buffer[i], m_devices[i]);
			guards.emplace_back(std::move(device_guard));
		}
		render_frame_main_multi_devices(m_devices, camera_matrix0, camera_matrix1, orig_screen_center, relative_focal_length, nerf_rolling_shutter, foveation, visualized_dimension);
	}
	cudaSetDevice(0);
	

	j4(m_devices[0].stream(), camera_matrix0, prev_camera_matrix, orig_screen_center, relative_focal_length, foveation, prev_foveation, render_buffer[0], to_srgb);
	guards.clear();
}


void NerfJobs::autofocus() {
	float new_slice_plane_z = std::max(dot(view_dir(), m_autofocus_target - view_pos()), 0.1f) - m_scale;
	if (new_slice_plane_z != m_slice_plane_z) {
		m_slice_plane_z = new_slice_plane_z;
		if (m_aperture_size != 0.0f) {
			reset_accumulation();
		}
	}
}

void NerfJobs::load_nerf_post() { // moved the second half of load_nerf here
	m_nerf.rgb_activation = m_nerf.training.dataset.is_hdr ? ENerfActivation::Exponential : ENerfActivation::Logistic;

	m_nerf.training.n_images_for_training = (int)m_nerf.training.dataset.n_images;

	m_nerf.training.dataset.update_metadata();

	m_nerf.training.cam_pos_gradient.resize(m_nerf.training.dataset.n_images, vec3(0.0f));
	m_nerf.training.cam_pos_gradient_gpu.resize_and_copy_from_host(m_nerf.training.cam_pos_gradient);

	m_nerf.training.cam_exposure.resize(m_nerf.training.dataset.n_images, AdamOptimizer<vec3>(1e-3f));
	m_nerf.training.cam_pos_offset.resize(m_nerf.training.dataset.n_images, AdamOptimizer<vec3>(1e-4f));
	m_nerf.training.cam_rot_offset.resize(m_nerf.training.dataset.n_images, RotationAdamOptimizer(1e-4f));
	m_nerf.training.cam_focal_length_offset = AdamOptimizer<vec2>(1e-5f);

	m_nerf.training.cam_rot_gradient.resize(m_nerf.training.dataset.n_images, vec3(0.0f));
	m_nerf.training.cam_rot_gradient_gpu.resize_and_copy_from_host(m_nerf.training.cam_rot_gradient);

	m_nerf.training.cam_exposure_gradient.resize(m_nerf.training.dataset.n_images, vec3(0.0f));
	m_nerf.training.cam_exposure_gpu.resize_and_copy_from_host(m_nerf.training.cam_exposure_gradient);
	m_nerf.training.cam_exposure_gradient_gpu.resize_and_copy_from_host(m_nerf.training.cam_exposure_gradient);

	m_nerf.training.cam_focal_length_gradient = vec2(0.0f);
	m_nerf.training.cam_focal_length_gradient_gpu.resize_and_copy_from_host(&m_nerf.training.cam_focal_length_gradient, 1);

	m_nerf.training.reset_extra_dims(m_rng);
	m_nerf.training.optimize_extra_dims = m_nerf.training.dataset.n_extra_learnable_dims > 0;

	if (m_nerf.training.dataset.has_rays) {
		m_nerf.training.near_distance = 0.0f;
	}

	// Perturbation of the training cameras -- for debugging the online extrinsics learning code
	// float perturb_amount = 0.0f;
	// if (perturb_amount > 0.f) {
	// 	for (uint32_t i = 0; i < m_nerf.training.dataset.n_images; ++i) {
	// 		vec3 rot = random_val_3d(m_rng) * perturb_amount;
	// 		float angle = rot.norm();
	// 		rot /= angle;
	// 		auto trans = random_val_3d(m_rng);
	// 		m_nerf.training.dataset.xforms[i].start.block<3,3>(0,0) = AngleAxisf(angle, rot).matrix() * m_nerf.training.dataset.xforms[i].start.block<3,3>(0,0);
	// 		m_nerf.training.dataset.xforms[i].start[3] += trans * perturb_amount;
	// 		m_nerf.training.dataset.xforms[i].end.block<3,3>(0,0) = AngleAxisf(angle, rot).matrix() * m_nerf.training.dataset.xforms[i].end.block<3,3>(0,0);
	// 		m_nerf.training.dataset.xforms[i].end[3] += trans * perturb_amount;
	// 	}
	// }

	m_nerf.training.update_transforms();

	if (!m_nerf.training.dataset.metadata.empty()) {
		m_nerf.render_lens = m_nerf.training.dataset.metadata[0].lens;
		m_screen_center = vec2(1.f) - m_nerf.training.dataset.metadata[0].principal_point;
	}

	if (!is_pot(m_nerf.training.dataset.aabb_scale)) {
		throw std::runtime_error{fmt::format("NeRF dataset's `aabb_scale` must be a power of two, but is {}.", m_nerf.training.dataset.aabb_scale)};
	}

	int max_aabb_scale = 1 << (NERF_CASCADES()-1);
	if (m_nerf.training.dataset.aabb_scale > max_aabb_scale) {
		throw std::runtime_error{fmt::format(
			"NeRF dataset must have `aabb_scale <= {}`, but is {}. "
			"You can increase this limit by factors of 2 by incrementing `NERF_CASCADES()` and re-compiling.",
			max_aabb_scale, m_nerf.training.dataset.aabb_scale
		)};
	}

	m_aabb = BoundingBox{vec3(0.5f), vec3(0.5f)};
	m_aabb.inflate(0.5f * std::min(1 << (NERF_CASCADES()-1), m_nerf.training.dataset.aabb_scale));
	m_raw_aabb = m_aabb;
	m_render_aabb = m_aabb;
	m_render_aabb_to_local = m_nerf.training.dataset.render_aabb_to_local;
	if (!m_nerf.training.dataset.render_aabb.is_empty()) {
		m_render_aabb = m_nerf.training.dataset.render_aabb.intersection(m_aabb);
	}

	m_nerf.max_cascade = 0;
	while ((1 << m_nerf.max_cascade) < m_nerf.training.dataset.aabb_scale) {
		++m_nerf.max_cascade;
	}

	// Perform fixed-size stepping in unit-cube scenes (like original NeRF) and exponential
	// stepping in larger scenes.
	m_nerf.cone_angle_constant = m_nerf.training.dataset.aabb_scale <= 1 ? 0.0f : (1.0f / 256.0f);

	m_up_dir = m_nerf.training.dataset.up;
}

void NerfJobs::load_nerf(const fs::path& data_path) {
	if (!data_path.empty()) {
		std::vector<fs::path> json_paths;
		if (data_path.is_directory()) {
			for (const auto& path : fs::directory{data_path}) {
				if (path.is_file() && equals_case_insensitive(path.extension(), "json")) {
					json_paths.emplace_back(path);
				}
			}
		} else if (equals_case_insensitive(data_path.extension(), "json")) {
			json_paths.emplace_back(data_path);
		} else {
			throw std::runtime_error{"NeRF data path must either be a json file or a directory containing json files."};
		}

		const auto prev_aabb_scale = m_nerf.training.dataset.aabb_scale;

		m_nerf.training.dataset = ngp::load_nerf(json_paths, m_nerf.sharpen);

		// Check if the NeRF network has been previously configured.
		// If it has not, don't reset it.
		if (m_nerf.training.dataset.aabb_scale != prev_aabb_scale && m_nerf_network) {
			// The AABB scale affects network size indirectly. If it changed after loading,
			// we need to reset the previously configured network to keep a consistent internal state.
			reset_network();
		}
	}

	load_nerf_post();
}


NerfJobs::NetworkDims NerfJobs::network_dims() const {
	switch (m_testbed_mode) {
		case ETestbedMode::Nerf:   return network_dims_nerf(); break;
		// case ETestbedMode::Sdf:    return network_dims_sdf(); break;
		case ETestbedMode::Image:  return network_dims_image(); break;
		// case ETestbedMode::Volume: return network_dims_volume(); break;
		default: throw std::runtime_error{"Invalid mode."};
	}
}

ELossType NerfJobs::string_to_loss_type(const std::string& str) {
	if (equals_case_insensitive(str, "L2")) {
		return ELossType::L2;
	} else if (equals_case_insensitive(str, "RelativeL2")) {
		return ELossType::RelativeL2;
	} else if (equals_case_insensitive(str, "L1")) {
		return ELossType::L1;
	} else if (equals_case_insensitive(str, "Mape")) {
		return ELossType::Mape;
	} else if (equals_case_insensitive(str, "Smape")) {
		return ELossType::Smape;
	} else if (equals_case_insensitive(str, "Huber") || equals_case_insensitive(str, "SmoothL1")) {
		// Legacy: we used to refer to the Huber loss (L2 near zero, L1 further away) as "SmoothL1".
		return ELossType::Huber;
	} else if (equals_case_insensitive(str, "LogL1")) {
		return ELossType::LogL1;
	} else {
		throw std::runtime_error{"Unknown loss type."};
	}
}

NerfJobs::NetworkDims NerfJobs::network_dims_nerf() const {
	NetworkDims dims;
	dims.n_input = sizeof(NerfCoordinate) / sizeof(float);
	dims.n_output = 4;
	dims.n_pos = sizeof(NerfPosition) / sizeof(float);
	return dims;
}


void NerfJobs::Nerf::Training::reset_extra_dims(default_rng_t& rng) {
	uint32_t n_extra_dims = dataset.n_extra_dims();
	std::vector<float> extra_dims_cpu(n_extra_dims * (dataset.n_images + 1)); // n_images + 1 since we use an extra 'slot' for the inference latent code
	float* dst = extra_dims_cpu.data();
	extra_dims_opt = std::vector<VarAdamOptimizer>(dataset.n_images, VarAdamOptimizer(n_extra_dims, 1e-4f));
	for (uint32_t i = 0; i < dataset.n_images; ++i) {
		vec3 light_dir = warp_direction_jobs(normalize(dataset.metadata[i].light_dir));
		extra_dims_opt[i].reset_state();
		std::vector<float>& optimzer_value = extra_dims_opt[i].variable();
		for (uint32_t j = 0; j < n_extra_dims; ++j) {
			if (dataset.has_light_dirs && j < 3) {
				dst[j] = light_dir[j];
			} else {
				dst[j] = random_val(rng) * 2.0f - 1.0f;
			}
			optimzer_value[j] = dst[j];
		}
		dst += n_extra_dims;
	}
	extra_dims_gpu.resize_and_copy_from_host(extra_dims_cpu);
}

void NerfJobs::CudaDevice::set_nerf_network(const std::shared_ptr<NerfNetwork<precision_t>>& nerf_network) {
	m_network = m_nerf_network = nerf_network;
}

void NerfJobs::reset_network(bool clear_density_grid) {
	m_sdf.iou_decay = 0;

	m_rng = default_rng_t{m_seed};

	// Start with a low rendering resolution and gradually ramp up
	m_render_ms.set(10000);

	reset_accumulation();
	m_nerf.training.counters_rgb.rays_per_batch = 1 << 12;
	m_nerf.training.counters_rgb.measured_batch_size_before_compaction = 0;
	m_nerf.training.n_steps_since_cam_update = 0;
	m_nerf.training.n_steps_since_error_map_update = 0;
	m_nerf.training.n_rays_since_error_map_update = 0;
	m_nerf.training.n_steps_between_error_map_updates = 128;
	m_nerf.training.error_map.is_cdf_valid = false;
	m_nerf.training.density_grid_rng = default_rng_t{m_rng.next_uint()};

	m_nerf.training.reset_camera_extrinsics();

	if (clear_density_grid) {
		m_nerf.density_grid.memset(0);
		m_nerf.density_grid_bitfield.memset(0);

		set_all_devices_dirty();
	}

	m_loss_graph_samples = 0;

	// Default config
	json config = m_network_config;

	json& encoding_config = config["encoding"];
	json& loss_config = config["loss"];
	json& optimizer_config = config["optimizer"];
	json& network_config = config["network"];

	// If the network config is incomplete, avoid doing further work.
	/*
	if (config.is_null() || encoding_config.is_null() || loss_config.is_null() || optimizer_config.is_null() || network_config.is_null()) {
		return;
	}
	*/

	auto dims = network_dims();

	if (m_testbed_mode == ETestbedMode::Nerf) {
		m_nerf.training.loss_type = string_to_loss_type(loss_config.value("otype", "L2"));

		// Some of the Nerf-supported losses are not supported by tcnn::Loss,
		// so just create a dummy L2 loss there. The NeRF code path will bypass
		// the tcnn::Loss in any case.
		loss_config["otype"] = "L2";
	}

	// Automatically determine certain parameters if we're dealing with the (hash)grid encoding
	if (to_lower(encoding_config.value("otype", "OneBlob")).find("grid") != std::string::npos) {
		encoding_config["n_pos_dims"] = dims.n_pos;

		m_n_features_per_level = encoding_config.value("n_features_per_level", 2u);

		if (encoding_config.contains("n_features") && encoding_config["n_features"] > 0) {
			m_n_levels = (uint32_t)encoding_config["n_features"] / m_n_features_per_level;
		} else {
			m_n_levels = encoding_config.value("n_levels", 16u);
		}

		m_level_stats.resize(m_n_levels);
		m_first_layer_column_stats.resize(m_n_levels);

		const uint32_t log2_hashmap_size = encoding_config.value("log2_hashmap_size", 15);

		m_base_grid_resolution = encoding_config.value("base_resolution", 0);
		if (!m_base_grid_resolution) {
			m_base_grid_resolution = 1u << ((log2_hashmap_size) / dims.n_pos);
			encoding_config["base_resolution"] = m_base_grid_resolution;
		}

		float desired_resolution = 2048.0f; // Desired resolution of the finest hashgrid level over the unit cube
		if (m_testbed_mode == ETestbedMode::Image) {
			desired_resolution = compMax(m_image.resolution) / 2.0f;
		} else if (m_testbed_mode == ETestbedMode::Volume) {
			desired_resolution = m_volume.world2index_scale;
		}

		// Automatically determine suitable per_level_scale
		m_per_level_scale = encoding_config.value("per_level_scale", 0.0f);
		if (m_per_level_scale <= 0.0f && m_n_levels > 1) {
			m_per_level_scale = std::exp(std::log(desired_resolution * (float)m_nerf.training.dataset.aabb_scale / (float)m_base_grid_resolution) / (m_n_levels-1));
			encoding_config["per_level_scale"] = m_per_level_scale;
		}

		tlog::info()
			<< "GridEncoding: "
			<< " Nmin=" << m_base_grid_resolution
			<< " b=" << m_per_level_scale
			<< " F=" << m_n_features_per_level
			<< " T=2^" << log2_hashmap_size
			<< " L=" << m_n_levels
			;
	}

	m_loss.reset(create_loss<precision_t>(loss_config));
	m_optimizer.reset(create_optimizer<precision_t>(optimizer_config));

	size_t n_encoding_params = 0;
	if (m_testbed_mode == ETestbedMode::Nerf) {
		m_nerf.training.cam_exposure.resize(m_nerf.training.dataset.n_images, AdamOptimizer<vec3>(1e-3f));
		m_nerf.training.cam_pos_offset.resize(m_nerf.training.dataset.n_images, AdamOptimizer<vec3>(1e-4f));
		m_nerf.training.cam_rot_offset.resize(m_nerf.training.dataset.n_images, RotationAdamOptimizer(1e-4f));
		m_nerf.training.cam_focal_length_offset = AdamOptimizer<vec2>(1e-5f);

		m_nerf.training.reset_extra_dims(m_rng);

		json& dir_encoding_config = config["dir_encoding"];
		json& rgb_network_config = config["rgb_network"];

		uint32_t n_dir_dims = 3;
		uint32_t n_extra_dims = m_nerf.training.dataset.n_extra_dims();

		// Instantiate an additional model for each auxiliary GPU
		for (auto& device : m_devices) {
			device.set_nerf_network(std::make_shared<NerfNetwork<precision_t>>(
				dims.n_pos,
				n_dir_dims,
				n_extra_dims,
				dims.n_pos + 1, // The offset of 1 comes from the dt member variable of NerfCoordinate. HACKY
				encoding_config,
				dir_encoding_config,
				network_config,
				rgb_network_config
			));
		}

		m_network = m_nerf_network = primary_device().nerf_network();

		m_encoding = m_nerf_network->pos_encoding();
		n_encoding_params = m_encoding->n_params() + m_nerf_network->dir_encoding()->n_params();

		tlog::info()
			<< "Density model: " << dims.n_pos
			<< "--[" << std::string(encoding_config["otype"])
			<< "]-->" << m_nerf_network->pos_encoding()->padded_output_width()
			<< "--[" << std::string(network_config["otype"])
			<< "(neurons=" << (int)network_config["n_neurons"] << ",layers=" << ((int)network_config["n_hidden_layers"]+2) << ")"
			<< "]-->" << 1
			;

		tlog::info()
			<< "Color model:   " << n_dir_dims
			<< "--[" << std::string(dir_encoding_config["otype"])
			<< "]-->" << m_nerf_network->dir_encoding()->padded_output_width() << "+" << network_config.value("n_output_dims", 16u)
			<< "--[" << std::string(rgb_network_config["otype"])
			<< "(neurons=" << (int)rgb_network_config["n_neurons"] << ",layers=" << ((int)rgb_network_config["n_hidden_layers"]+2) << ")"
			<< "]-->" << 3
			;


		// Create distortion map model
		{
			json& distortion_map_optimizer_config =  config.contains("distortion_map") && config["distortion_map"].contains("optimizer") ? config["distortion_map"]["optimizer"] : optimizer_config;

			m_distortion.resolution = ivec2(32);
			if (config.contains("distortion_map") && config["distortion_map"].contains("resolution")) {
				from_json(config["distortion_map"]["resolution"], m_distortion.resolution);
			}
			m_distortion.map = std::make_shared<TrainableBuffer<2, 2, float>>(m_distortion.resolution);
			m_distortion.optimizer.reset(create_optimizer<float>(distortion_map_optimizer_config));
			m_distortion.trainer = std::make_shared<Trainer<float, float>>(m_distortion.map, m_distortion.optimizer, std::shared_ptr<Loss<float>>{create_loss<float>(loss_config)}, m_seed);
		}
	} 
	size_t n_network_params = m_network->n_params() - n_encoding_params;

	tlog::info() << "  total_encoding_params=" << n_encoding_params << " total_network_params=" << n_network_params;

	m_trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(m_network, m_optimizer, m_loss, m_seed);
	m_training_step = 0;
	m_training_start_time_point = std::chrono::steady_clock::now();

	// Create envmap model
	{
		json& envmap_loss_config = config.contains("envmap") && config["envmap"].contains("loss") ? config["envmap"]["loss"] : loss_config;
		json& envmap_optimizer_config =  config.contains("envmap") && config["envmap"].contains("optimizer") ? config["envmap"]["optimizer"] : optimizer_config;

		m_envmap.loss_type = string_to_loss_type(envmap_loss_config.value("otype", "L2"));

		m_envmap.resolution = m_nerf.training.dataset.envmap_resolution;
		m_envmap.envmap = std::make_shared<TrainableBuffer<4, 2, float>>(m_envmap.resolution);
		m_envmap.optimizer.reset(create_optimizer<float>(envmap_optimizer_config));
		m_envmap.trainer = std::make_shared<Trainer<float, float, float>>(m_envmap.envmap, m_envmap.optimizer, std::shared_ptr<Loss<float>>{create_loss<float>(envmap_loss_config)}, m_seed);

		if (m_nerf.training.dataset.envmap_data.data()) {
			m_envmap.trainer->set_params_full_precision(m_nerf.training.dataset.envmap_data.data(), m_nerf.training.dataset.envmap_data.size());
		}
	}

	set_all_devices_dirty();
}

void NerfJobs::set_camera_from_keyframe(const CameraKeyframe& k) {
	m_camera = k.m();
	m_slice_plane_z = k.slice;
	m_scale = k.scale;
	set_fov(k.fov);
	m_aperture_size = k.aperture_size;
	m_nerf.glow_mode = k.glow_mode;
	m_nerf.glow_y_cutoff = k.glow_y_cutoff;
}


void NerfJobs::clear_training_data() {
	m_training_data_available = false;
	m_nerf.training.dataset.metadata.clear();
}

float NerfJobs::fov() const {
	return focal_length_to_fov(1.0f, m_relative_focal_length[m_fov_axis]);
}

void NerfJobs::set_fov(float val) {
	m_relative_focal_length = vec2(fov_to_focal_length(1, val));
}

vec2 NerfJobs::fov_xy() const {
	return focal_length_to_fov(ivec2(1), m_relative_focal_length);
}

void NerfJobs::set_fov_xy(const vec2& val) {
	m_relative_focal_length = fov_to_focal_length(ivec2(1), val);
}

void NerfJobs::reset_camera() {
	m_fov_axis = 1;
	m_zoom = 1.0f;
	m_screen_center = vec2(0.5f);

	if (m_testbed_mode == ETestbedMode::Image) {
		// Make image full-screen at the given view distance
		m_relative_focal_length = vec2(1.0f);
		m_scale = 1.0f;
	} else {
		set_fov(50.625f);
		m_scale = 1.5f;
	}

	m_camera = transpose(mat3x4(
		1.0f, 0.0f, 0.0f, 0.5f,
		0.0f, -1.0f, 0.0f, 0.5f,
		0.0f, 0.0f, -1.0f, 0.5f
	));

	m_camera[3] -= m_scale * view_dir();

	m_smoothed_camera = m_camera;
	m_sun_dir = normalize(vec3(1.0f));

	reset_accumulation();
}

void NerfJobs::set_mode(ETestbedMode mode) {
	if (mode == m_testbed_mode) {
		return;
	}

	// Reset mode-specific members
	m_image = {};
	m_mesh = {};
	m_nerf = {};
	m_sdf = {};
	m_volume = {};

	// Kill training-related things
	m_encoding = {};
	m_loss = {};
	m_network = {};
	m_nerf_network = {};
	m_optimizer = {};
	m_trainer = {};
	m_envmap = {};
	m_distortion = {};
	m_training_data_available = false;

	// Clear device-owned data that might be mode-specific
	for (auto&& device : m_devices) {
		device.clear();
	}

	// Reset paths that might be attached to the chosen mode
	m_data_path = {};

	m_testbed_mode = mode;

	// Set various defaults depending on mode
	if (m_testbed_mode == ETestbedMode::Nerf) {
		if (m_devices.size() > 1) {
			m_use_aux_devices = true;
		}

		if (m_dlss_provider && m_aperture_size == 0.0f) {
			m_dlss = true;
		}
	} else {
		m_use_aux_devices = false;
		m_dlss = false;
	}

	reset_camera();

#ifdef NGP_GUI
	update_vr_performance_settings();
#endif
}

void NerfJobs::reset_accumulation(bool due_to_camera_movement, bool immediate_redraw) {
	if (immediate_redraw) {
		redraw_next_frame();
	}

	if (!due_to_camera_movement || !reprojection_available()) {
		m_windowless_render_surface.reset_accumulation();
		for (auto& view : m_views) {
			view.render_buffer->reset_accumulation();
		}
	}
}

void NerfJobs::set_camera_from_time(float t) {
	if (m_camera_path.keyframes.empty()) {
		return;
	}

	set_camera_from_keyframe(m_camera_path.eval_camera_path(t));
}
static const size_t SNAPSHOT_FORMAT_VERSION = 1;

void NerfJobs::load_snapshot(const fs::path& path) {
	auto config = load_network_config(path);
	if (!config.contains("snapshot")) {
		throw std::runtime_error{fmt::format("File '{}' does not contain a snapshot.", path.str())};
	}

	const auto& snapshot = config["snapshot"];
	if (snapshot.value("version", 0) < SNAPSHOT_FORMAT_VERSION) {
		throw std::runtime_error{"Snapshot uses an old format and can not be loaded."};
	}

	if (snapshot.contains("mode")) {
		set_mode(mode_from_string(snapshot["mode"]));
	} else if (snapshot.contains("nerf")) {
		// To be able to load old NeRF snapshots that don't specify their mode yet
		set_mode(ETestbedMode::Nerf);
	} else if (m_testbed_mode == ETestbedMode::None) {
		throw std::runtime_error{"Unknown snapshot mode. Snapshot must be regenerated with a new version of instant-ngp."};
	}

	m_aabb = snapshot.value("aabb", m_aabb);
	m_bounding_radius = snapshot.value("bounding_radius", m_bounding_radius);

	if (m_testbed_mode == ETestbedMode::Nerf) {
		if (snapshot["density_grid_size"] != NERF_GRIDSIZE()) {
			throw std::runtime_error{"Incompatible grid size."};
		}

		m_nerf.training.counters_rgb.rays_per_batch = snapshot["nerf"]["rgb"]["rays_per_batch"];
		m_nerf.training.counters_rgb.measured_batch_size = snapshot["nerf"]["rgb"]["measured_batch_size"];
		m_nerf.training.counters_rgb.measured_batch_size_before_compaction = snapshot["nerf"]["rgb"]["measured_batch_size_before_compaction"];

		// If we haven't got a nerf dataset loaded, load dataset metadata from the snapshot
		// and render using just that.
		if (m_data_path.empty() && snapshot["nerf"].contains("dataset")) {
			m_nerf.training.dataset = snapshot["nerf"]["dataset"];
			load_nerf(m_data_path);
		} else {
			if (snapshot["nerf"].contains("aabb_scale")) {
				m_nerf.training.dataset.aabb_scale = snapshot["nerf"]["aabb_scale"];
			}

			if (snapshot["nerf"].contains("dataset")) {
				m_nerf.training.dataset.n_extra_learnable_dims = snapshot["nerf"]["dataset"].value("n_extra_learnable_dims", m_nerf.training.dataset.n_extra_learnable_dims);
			}
		}

		load_nerf_post();

		GPUMemory<__half> density_grid_fp16 = snapshot["density_grid_binary"];
		m_nerf.density_grid.resize(density_grid_fp16.size());

		parallel_for_gpu(density_grid_fp16.size(), [density_grid=m_nerf.density_grid.data(), density_grid_fp16=density_grid_fp16.data()] __device__ (size_t i) {
			density_grid[i] = (float)density_grid_fp16[i];
		});

		if (m_nerf.density_grid.size() == NERF_GRID_N_CELLS() * (m_nerf.max_cascade + 1)) {
			update_density_grid_mean_and_bitfield(nullptr);
		} else if (m_nerf.density_grid.size() != 0) {
			// A size of 0 indicates that the density grid was never populated, which is a valid state of a (yet) untrained model.
			throw std::runtime_error{"Incompatible number of grid cascades."};
		}
	}

	// Needs to happen after `load_nerf_post()`
	m_sun_dir = snapshot.value("sun_dir", m_sun_dir);
	m_exposure = snapshot.value("exposure", m_exposure);

#ifdef NGP_GUI
	if (!m_hmd)
#endif
	m_background_color = snapshot.value("background_color", m_background_color);

	if (snapshot.contains("camera")) {
		m_camera = snapshot["camera"].value("matrix", m_camera);
		m_fov_axis = snapshot["camera"].value("fov_axis", m_fov_axis);
		if (snapshot["camera"].contains("relative_focal_length")) from_json(snapshot["camera"]["relative_focal_length"], m_relative_focal_length);
		if (snapshot["camera"].contains("screen_center")) from_json(snapshot["camera"]["screen_center"], m_screen_center);
		m_zoom = snapshot["camera"].value("zoom", m_zoom);
		m_scale = snapshot["camera"].value("scale", m_scale);

		m_aperture_size = snapshot["camera"].value("aperture_size", m_aperture_size);
		if (m_aperture_size != 0) {
			m_dlss = false;
		}

		m_autofocus = snapshot["camera"].value("autofocus", m_autofocus);
		if (snapshot["camera"].contains("autofocus_target")) from_json(snapshot["camera"]["autofocus_target"], m_autofocus_target);
		m_slice_plane_z = snapshot["camera"].value("autofocus_depth", m_slice_plane_z);
	}

	if (snapshot.contains("render_aabb_to_local")) from_json(snapshot.at("render_aabb_to_local"), m_render_aabb_to_local);
	m_render_aabb = snapshot.value("render_aabb", m_render_aabb);
	if (snapshot.contains("up_dir")) from_json(snapshot.at("up_dir"), m_up_dir);

	m_network_config_path = path;
	m_network_config = std::move(config);

	reset_network(false);

	m_training_step = m_network_config["snapshot"]["training_step"];
	m_loss_scalar.set(m_network_config["snapshot"]["loss"]);

	m_trainer->deserialize(m_network_config["snapshot"]);

	if (m_testbed_mode == ETestbedMode::Nerf) {
		// If the snapshot appears to come from the same dataset as was already present
		// (or none was previously present, in which case it came from the snapshot
		// in the first place), load dataset-specific optimized quantities, such as
		// extrinsics, exposure, latents.
		if (snapshot["nerf"].contains("dataset") && m_nerf.training.dataset.is_same(snapshot["nerf"]["dataset"])) {
			if (snapshot["nerf"].contains("cam_pos_offset")) m_nerf.training.cam_pos_offset = snapshot["nerf"].at("cam_pos_offset").get<std::vector<AdamOptimizer<vec3>>>();
			if (snapshot["nerf"].contains("cam_rot_offset")) m_nerf.training.cam_rot_offset = snapshot["nerf"].at("cam_rot_offset").get<std::vector<RotationAdamOptimizer>>();
			if (snapshot["nerf"].contains("extra_dims_opt")) m_nerf.training.extra_dims_opt = snapshot["nerf"].at("extra_dims_opt").get<std::vector<VarAdamOptimizer>>();
			m_nerf.training.update_transforms();
			m_nerf.training.update_extra_dims();
		}
	}

	set_all_devices_dirty();
}

void NerfJobs::apply_camera_smoothing(float elapsed_ms) {
	if (m_camera_smoothing) {
		float decay = std::pow(0.02f, elapsed_ms/1000.0f);
		m_smoothed_camera = camera_lerp(m_smoothed_camera, m_camera, 1.0f - decay);
	} else {
		m_smoothed_camera = m_camera;
	}
}

int NerfJobs::render_to_test(int width, int height, int spp, bool linear, float start_time, float end_time, float fps, float shutter_fraction) {
	cudaSetDevice(0);
	std::vector<CudaRenderBuffer> m_windowless_render_surfaces;
	std::vector<std::shared_ptr<CudaSurface2D>> surfaces;
	
	for(int i = 0;i< m_devices.size();i++){
		std::shared_ptr<CudaSurface2D> surface = std::make_shared<CudaSurface2D>();
		surfaces.push_back(surface);
		CudaRenderBuffer buffer{surface};
		buffer.resize({width, height});
		buffer.reset_accumulation();
		m_windowless_render_surfaces.push_back(std::move(buffer));
	}
	
	m_windowless_render_surface.resize({width, height});
	m_windowless_render_surface.reset_accumulation();

	if (end_time < 0.f) {
		end_time = start_time;
	}
	bool path_animation_enabled = start_time >= 0.f;
	if (!path_animation_enabled) { // the old code disabled camera smoothing for non-path renders; so we preserve that behaviour
		m_smoothed_camera = m_camera;
	}
	// this rendering code assumes that the intra-frame camera motion starts from m_smoothed_camera (ie where we left off) to allow for EMA camera smoothing.
	// in the case of a camera path animation, at the very start of the animation, we have yet to initialize smoothed_camera to something sensible
	// - it will just be the default boot position. oops!
	// that led to the first frame having a crazy streak from the default camera position to the start of the path.
	// so we detect that case and explicitly force the current matrix to the start of the path
	if (start_time == 0.f) {
		set_camera_from_time(start_time);
		m_smoothed_camera = m_camera;
	}

	auto start_cam_matrix = m_smoothed_camera;

	// now set up the end-of-frame camera matrix if we are moving along a path
	if (path_animation_enabled) {
		set_camera_from_time(end_time);
		apply_camera_smoothing(1000.f / fps);
	}

	auto end_cam_matrix = m_smoothed_camera;
	auto prev_camera_matrix = m_smoothed_camera;

	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < spp; ++i) {
		float start_alpha = ((float)i)/(float)spp * shutter_fraction;
		float end_alpha = ((float)i + 1.0f)/(float)spp * shutter_fraction;

		auto sample_start_cam_matrix = start_cam_matrix;
		auto sample_end_cam_matrix = camera_lerp(start_cam_matrix, end_cam_matrix, shutter_fraction);
		if (i == 0) {
			prev_camera_matrix = sample_start_cam_matrix;
		}

		if (path_animation_enabled) {
			set_camera_from_time(start_time + (end_time-start_time) * (start_alpha + end_alpha) / 2.0f);
			m_smoothed_camera = m_camera;
		}

		if (m_autofocus) {
			autofocus();
		}

		render_frame_multi_devices(
			sample_start_cam_matrix,
			sample_end_cam_matrix,
			prev_camera_matrix,
			m_screen_center,
			m_relative_focal_length,
			{0.0f, 0.0f, 0.0f, 1.0f},
			{},
			{},
			m_visualized_dimension,
			m_windowless_render_surfaces,
			!linear
		);
		prev_camera_matrix = sample_start_cam_matrix;
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	total_time += elapsed;
	std::cout << "Rendered " << spp << " samples in " << elapsed << "ms" << std::endl;
	std::cout << "total time " << total_time  << "ms" << std::endl;
	// For cam smoothing when rendering the next frame.
	m_smoothed_camera = end_cam_matrix;

	float *a = (float *)malloc(width * sizeof(float) * 4 *height);
	CUDA_CHECK_THROW(cudaMemcpy2DFromArray(a, width * sizeof(float) * 4, m_windowless_render_surfaces[0].surface_provider().array(), 0, 0, width * sizeof(float) * 4, height, cudaMemcpyDeviceToHost));
	FILE* file = fopen("result.bin", "wb");
    if (!file) {
        std::cout << "Failed to open file for writing: " << "result.bin" << std::endl;
        return 0;
    }
	std::cout << "width: " << width << " height: " << height << std::endl; 

    fwrite(a, sizeof(float), width * 4 * height, file);

    fclose(file);
	
	return true;
}


NGP_NAMESPACE_END
int main(int argc, char* argv[]) {
	if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <path to snapshot>" << std::endl;
        return 1;
    }
    
    ngp::NerfJobs nerfJobs;
    auto rootDir = filesystem::path::getcwd();
    auto testTransforms = argv[2];
    nerfJobs.m_root_dir = rootDir;
    nerfJobs.load_snapshot(argv[1]);

    nerfJobs.m_nerf.sharpen = 0.0f;
    nerfJobs.m_exposure = 0.0f;
    nerfJobs.m_train = false;

    nerfJobs.m_nerf.render_with_lens_distortion = true;

    std::cout << "Evaluating test transforms from " << testTransforms << std::endl;

    float totmse = 0.0f;
    float totpsnr = 0.0f;
    float totssim = 0.0f;
    float totcount = 0.0f;
    float minpsnr = 1000.0f;
    float maxpsnr = 0.0f;

    nerfJobs.m_background_color = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    nerfJobs.m_snap_to_pixel_centers = true;
    int spp = 1;
    nerfJobs.m_nerf.render_min_transmittance = 1e-4f;

    nerfJobs.m_train = false;
    nerfJobs.load_training_data(testTransforms);
    auto t = nerfJobs.m_nerf.training.dataset.n_images;
	for (int i = 0; i < t; i++){
        auto resolution = nerfJobs.m_nerf.training.dataset.metadata[i].resolution;
        nerfJobs.m_render_ground_truth = false;
        nerfJobs.set_camera_to_training_view(i);

        nerfJobs.render_to_test(resolution.x, resolution.y, spp, false,-1.0f,-1.0f,30.0f,1.0f);
    }
}
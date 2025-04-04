#include <project.hpp>
#include <postprocess.hpp>
#include <netTensorRT.hpp>
#include <NvInferVersion.h>
#include <pcl/visualization/point_cloud_geometry_handlers.h>
#include <pcl/visualization/impl/point_cloud_geometry_handlers.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <fmt/core.h>

namespace rangenet
{
  namespace segmentation
  {
    NetTensorRT::NetTensorRT(const std::string& model_path, bool use_pcl_viewer)
      : Net(model_path), use_pcl_viewer_(use_pcl_viewer)
    {
      std::string onnx_path = model_path + "model.onnx";
      std::string engine_path = model_path + fmt::format("model_{}.{}.{}.trt", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);

      std::cout << "Trying to open model" << std::endl;
      std::fstream file(engine_path, std::ios::binary | std::ios::in);
      if (!file.is_open())
      {
        std::cout << "read engine file " << engine_path << " failed" << std::endl;
        std::cout << "Could not deserialize TensorRT engine. " << std::endl
          << "Generating from sratch... This may take a while..."
          << std::endl;
        serializeEngine(onnx_path, engine_path);
      }

      deserializeEngine(engine_path);
      prepareBuffer();
      CHECK_CUDA_ERROR(cudaEventCreate(&start_));
      CHECK_CUDA_ERROR(cudaEventCreate(&stop_));
      CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
    }

    NetTensorRT::~NetTensorRT()
    {
      // free cuda buffers
      CHECK_CUDA_ERROR(cudaFree(device_buffers_[1]));
      std::cout << "cuda buffers released." << std::endl;
      // free cuda pinned mem
      for (auto& buffer : host_buffers_)
        CHECK_CUDA_ERROR(cudaFreeHost(buffer));
      std::cout << "cuda pinned mem released." << std::endl;

      CHECK_CUDA_ERROR(cudaStreamDestroy(stream_));
      CHECK_CUDA_ERROR(cudaEventDestroy(start_));
      CHECK_CUDA_ERROR(cudaEventDestroy(stop_));
    }

    void NetTensorRT::doInfer(const pcl::PointCloud<PointType>& pointcloud_pcl,
                              int labels[])
    {
      uint32_t num_points = pointcloud_pcl.size();
      // check if engine is valid
      if (!engine_)
      {
        throw std::runtime_error("Invaild engine on inference.");
      }

#if PERFORMANCE_LOG
  float preprocess_time = 0.0f;
  CHECK_CUDA_ERROR(cudaEventRecord(start_, stream_));
#endif

      // step1: generate input image
      ProjectGPU project_gpu(stream_);
      project_gpu.doProject(pointcloud_pcl, false);
      CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
      float* range_img_device = project_gpu.range_img_device_.get();
      device_buffers_[0] = range_img_device;

#if PERFORMANCE_LOG
  CHECK_CUDA_ERROR(cudaEventRecord(stop_, stream_));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop_));
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&preprocess_time, start_, stop_));
#endif

#if 0
  // opencv 可视化深度图
  float *range_img_cv;
  cudaMallocHost((void **)&range_img_cv,
                 FEATURE_DIMS * IMG_H * IMG_W * sizeof(float));

  cudaMemcpy(range_img_cv, range_img_device,
             FEATURE_DIMS * IMG_H * IMG_W * sizeof(float),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < FEATURE_DIMS; i++) {
    createColorImg(range_img_cv, i);
  }
  exit(0);
#endif

#if PERFORMANCE_LOG
  float infer_time = 0.0f;
  CHECK_CUDA_ERROR(cudaEventRecord(start_, stream_));
#endif

#if NV_TENSORRT_MAJOR >= 8 and NV_TENSORRT_MINOR >= 5
      auto const first_layer_name = engine_->getIOTensorName(0);
      auto const last_layer_name = engine_->getIOTensorName(engine_->getNbIOTensors() - 1);
      context_->setTensorAddress(first_layer_name, device_buffers_[0]);
      context_->setTensorAddress(last_layer_name, device_buffers_[1]);
      context_->enqueueV3(stream_);
#else
      context_->enqueueV2((void**)&device_buffers_[0], stream_, nullptr);
#endif

      CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

#if NV_TENSORRT_MAJOR >= 8 and NV_TENSORRT_MINOR >= 5
      auto layer_name = engine_->getIOTensorName(1);
      int totoal_size = getBufferSize(engine_->getTensorShape(layer_name),
                                     engine_->getTensorDataType(layer_name));
#else
      int totoal_size = getBufferSize(engine_->getBindingDimensions(1),
                                      engine_->getBindingDataType(1));
#endif

      CHECK_CUDA_ERROR(cudaMemcpy(host_buffers_[1], device_buffers_[1], totoal_size,
        cudaMemcpyDeviceToHost));
      CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

      float label_image[IMG_H * IMG_W];

      for (int pixel_id = 0; pixel_id < IMG_H * IMG_W; pixel_id++)
      {
        if (project_gpu.valid_idx_[pixel_id])
        {
          label_image[pixel_id] = ((int*)host_buffers_[1])[pixel_id];
        }
        else
        {
          label_image[pixel_id] = 0;
        }
      }

      // range image GPU->CPU
      CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
#if NV_TENSORRT_MAJOR >= 8 and NV_TENSORRT_MINOR >= 5
      layer_name = engine_->getIOTensorName(0);
      totoal_size = getBufferSize(engine_->getTensorShape(layer_name),
                                 engine_->getTensorDataType(layer_name));
#else
      totoal_size = getBufferSize(engine_->getBindingDimensions(0),
                                  engine_->getBindingDataType(0));
#endif

      CHECK_CUDA_ERROR(cudaMemcpy(host_buffers_[0], device_buffers_[0], totoal_size,
        cudaMemcpyDeviceToHost));
      CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

#if PERFORMANCE_LOG
  CHECK_CUDA_ERROR(cudaEventRecord(stop_, stream_));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop_));
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&infer_time, start_, stop_));
#endif

#if PERFORMANCE_LOG
  float postprocess_time = 0.0f;
  CHECK_CUDA_ERROR(cudaEventRecord(start_, stream_));
#endif

      // step3: postprocess
      bool isPostprocess = true;
      if (isPostprocess)
      {
        auto ptr = (float*)host_buffers_[0];
        // CHW
        float range_img[IMG_H * IMG_W];
        for (int pixel_x = 0; pixel_x < IMG_W; pixel_x++)
        {
          for (int pixel_y = 0; pixel_y < IMG_H; pixel_y++)
          {
            int pixel_id = 0 * IMG_H * IMG_W + pixel_y * IMG_W + pixel_x;
            range_img[pixel_id] = ptr[pixel_id] * 12.32 + 12.12;
          }
        }

        Postprocess postprocess(5, 1.0);
        postprocess.postprocessKNN(range_img, project_gpu.range_arr_.get(), label_image,
                                   project_gpu.pxs_.get(), project_gpu.pys_.get(),
                                   num_points, labels);
      }
      else
      {
        for (int i = 0; i < num_points; i++)
        {
          int pixel_x = project_gpu.pxs_[i];
          int pixel_y = project_gpu.pys_[i];
          int point_label = label_image[pixel_y * IMG_W + pixel_x];
          labels[i] = point_label;
        }
      }

#if PERFORMANCE_LOG
  CHECK_CUDA_ERROR(cudaEventRecord(stop_, stream_));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop_));
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&postprocess_time, start_, stop_));
  std::cout<<"TIME: preprocess_time: "<< preprocess_time  <<" ms." <<std::endl;
  std::cout<<"TIME: infer_time: "<< infer_time <<" ms." <<std::endl;
  std::cout<<"TIME: postprocess_time: "<< postprocess_time <<" ms." << std::endl;
#endif

      if (this->use_pcl_viewer_)
      {
        // Visualize the pointcloud
        pcl::PointCloud<pcl::PointXYZRGB> color_pointcloud;
        paintPointCloud(pointcloud_pcl, color_pointcloud, labels);
        pcl::visualization::PCLVisualizer viewer("3D Viewer");

#if VTK_MAJOR_VERSION == 9 && ((VTK_MINOR_VERSION == 0 && (VTK_BUILD_VERSION == 2 || VTK_BUILD_VERSION == 3)) || (VTK_MINOR_VERSION == 1 && VTK_BUILD_VERSION == 0))
        // Suppress the VTK warning
        viewer.getRenderWindow()->GlobalWarningDisplayOff();
#endif

        viewer.setBackgroundColor(0, 0, 0);
        viewer.addPointCloud<pcl::PointXYZRGB>(color_pointcloud.makeShared(), "sample cloud", 0);
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
        viewer.addCoordinateSystem(1.0);
        viewer.initCameraParameters();
        viewer.setCameraPosition(-29.8411, -5.16865, 9.70893, 0, 0, 0, 0, 0, 1);
        viewer.setCameraFieldOfView(0.523599);
        viewer.setCameraClipDistances(0.207283, 207.283);

        while (!viewer.wasStopped())
        {
          viewer.spinOnce();
        }
      }
    }

    /**
     * 获取输入/输出的需分配内存大小
     * @param d dimension
     * @param t type
     * @return
     */
    int NetTensorRT::getBufferSize(Dims d, DataType t)
    {
      int size = 1;
      for (int i = 0; i < d.nbDims; i++)
        size *= d.d[i];

      switch (t)
      {
      case DataType::kINT32: return size * 4;
      case DataType::kFLOAT: return size * 4;
      case DataType::kHALF: return size * 2;
      case DataType::kINT8: return size * 1;
      default: throw std::runtime_error("Data type not handled");
      }
      return 0;
    }

    /**
     * @brief Deserialize an engine that comes from a previous run
     *
     * @param engine_path
     */
    void NetTensorRT::deserializeEngine(const std::string& engine_path)
    {
      std::cout << "[INFO] Trying to deserialize previously stored: " << engine_path << std::endl;

      // Load engine
      std::fstream file(engine_path, std::ios::binary | std::ios::in);
      file.seekg(0, std::ios::end);
      int length = file.tellg();
      file.seekg(0, std::ios::beg);
      std::unique_ptr<char[]> data(new char[length]);
      file.read(data.get(), length);
      file.close();

      runtime_ = std::unique_ptr<IRuntime>(createInferRuntime(g_logger_));
      engine_ = std::unique_ptr<ICudaEngine>(runtime_->deserializeCudaEngine(data.get(), length));
      if (!engine_)
      {
        throw std::runtime_error("[ERROR] Invalid engine. The engine may not exist and should be created first.");
      }
      std::cout << "[INFO] Successfully deserialized engine from tensorrt file" << std::endl;
    }

    /**
     * @brief Serialize an engine that we generated in this run
     *
     * @param engine_path
     */
    void NetTensorRT::serializeEngine(const std::string& onnx_path,
                                      const std::string& engine_path)
    {
      g_logger_.log(Logger::Severity::kINFO, "TRYING TO SERIALIZE AND SAVE ENGINE");

      // Create inference builder
      auto builder = std::unique_ptr<IBuilder>(createInferBuilder(g_logger_));
      assert(builder != nullptr);
      const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
      auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());

      // const auto tacticType =
      //      1U << static_cast<uint32_t>(TacticSource::kCUBLAS) | 1U << static_cast<uint32_t>(TacticSource::kCUBLAS_LT)
      //          | 1U << static_cast<uint32_t>(TacticSource::kCUDNN);
      // config->setTacticSources(tacticType);

      config->setFlag(nvinfer1::BuilderFlag::kFP16);
#if NV_TENSORRT_MAJOR >= 8 and NV_TENSORRT_MINOR >= 5
      config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 5UL << 30);
#else
      config->setMaxWorkspaceSize(5UL << 30);
#endif
      config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);

      auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
      assert(network != nullptr);

      // Generate a parser to get weights from onnx file
      auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, g_logger_));
      parser->parseFromFile(onnx_path.c_str(), static_cast<int>(Logger::Severity::kWARNING));

      auto ktop_layer = network->addTopK(*network->getOutput(0), nvinfer1::TopKOperation::kMAX, 1, 2);
      assert(ktop_layer != nullptr);
      ktop_layer->setName("top_k_layer");
      ktop_layer->setPrecision(nvinfer1::DataType::kINT32);
      // Replace the original output of the network with the output of the topK layer
      network->unmarkOutput(*network->getOutput(0));
      network->markOutput(*ktop_layer->getOutput(1));

      auto plan = std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
      if (!plan)
      {
        throw std::runtime_error("Failed to build tensorrt engine");
      }

      std::ofstream planFile(engine_path, std::ios::binary);
      planFile.write(static_cast<char*>(plan->data()), plan->size());
    }

    void NetTensorRT::prepareBuffer()
    {
      context_ =
        std::unique_ptr<IExecutionContext>(engine_.get()->createExecutionContext());
      if (!context_)
      {
        throw std::runtime_error("Invalid execution context. Can't infer.");
      }

#if NV_TENSORRT_MAJOR >= 8 and NV_TENSORRT_MINOR >= 5
      int n_bindings = engine_->getNbIOTensors();
#else
      int n_bindings = engine_->getNbBindings();
#endif

      if (n_bindings != 2)
      {
        throw std::runtime_error("Invalid number of bindings: " +
          std::to_string(n_bindings));
      }

#if 0
  auto inspector =
      std::unique_ptr<IEngineInspector>(_engine->createEngineInspector());
  // 输出所有层的信息
  std::cout << inspector->getEngineInformation(
      LayerInformationFormat::kONELINE);
#endif

      // clear buffers and reserve memory
      device_buffers_.clear();
      device_buffers_.reserve(n_bindings);
      host_buffers_.clear();
      host_buffers_.reserve(n_bindings);

      // 分配显存
#if NV_TENSORRT_MAJOR >= 8 and NV_TENSORRT_MINOR >= 5
      int nbBindings = engine_->getNbIOTensors();
#else
      int nbBindings = engine_->getNbBindings();
#endif

      for (int i = 0; i < nbBindings; i++)
      {
#if NV_TENSORRT_MAJOR >= 8 and NV_TENSORRT_MINOR >= 5
        auto layer_name = engine_->getIOTensorName(i);
        auto dims = engine_->getTensorShape(layer_name);
        auto dtype = engine_->getTensorDataType(layer_name);
#else
        Dims dims = engine_->getBindingDimensions(i);
        DataType dtype = engine_->getBindingDataType(i);
#endif
        uint64_t totalSize = getBufferSize(dims, dtype);
        CHECK_CUDA_ERROR(cudaMalloc(&device_buffers_[i], totalSize));
        CHECK_CUDA_ERROR(cudaMallocHost(&host_buffers_[i], totalSize));

        std::string info = "I/O dimensions respectively are: [ ";
        for (int d = 0; d < dims.nbDims; d++)
        {
          info += std::to_string(dims.d[d]) + " ";
        }
        info = info + "]";
        g_logger_.log(nvinfer1::ILogger::Severity::kINFO, info.c_str());
      }
    }

    void NetTensorRT::paintPointCloud(
      const pcl::PointCloud<PointType>& pointcloud,
      pcl::PointCloud<pcl::PointXYZRGB>& color_pointcloud, int labels[])
    {
      pcl::PointXYZRGB point;
      int point_num = pointcloud.size();
      for (size_t i = 0; i < point_num; ++i)
      {
        point.x = pointcloud.points[i].x;
        point.y = pointcloud.points[i].y;
        point.z = pointcloud.points[i].z;

        int label = labels[i];
        uint32_t r = std::get<2>(_argmax_to_rgb[label]);
        uint32_t g = std::get<1>(_argmax_to_rgb[label]);
        uint32_t b = std::get<0>(_argmax_to_rgb[label]);

        uint32_t rgb = (r << 16) | (g << 8) | b;
        point.rgb = *reinterpret_cast<float*>(&rgb);
        color_pointcloud.push_back(point);
      }
    }
  } // namespace segmentation
} // namespace rangenet

#ifndef RAY_CORE_WORKER_COMMON_H
#define RAY_CORE_WORKER_COMMON_H

#include <string>

#include "ray/common/buffer.h"
#include "ray/common/id.h"
#include "ray/protobuf/gcs.pb.h"
#include "ray/raylet/raylet_client.h"
#include "ray/raylet/task_spec.h"

namespace ray {

/// Type of this worker.
enum class WorkerType { WORKER, DRIVER };

/// Information about a remote function.
struct RayFunction {
  /// Language of the remote function.
  const ray::rpc::Language language;
  /// Function descriptor of the remote function.
  const std::vector<std::string> function_descriptor;
};

/// Argument of a task.
class TaskArg {
 public:
  /// Create a pass-by-reference task argument.
  ///
  /// \param[in] object_id Id of the argument.
  /// \return The task argument.
  static TaskArg PassByReference(const ObjectID &object_id) {
    return TaskArg(std::make_shared<ObjectID>(object_id), nullptr);
  }

  /// Create a pass-by-reference task argument.
  ///
  /// \param[in] object_id Id of the argument.
  /// \return The task argument.
  static TaskArg PassByValue(const std::shared_ptr<Buffer> &data) {
    return TaskArg(nullptr, data);
  }

  /// Return true if this argument is passed by reference, false if passed by value.
  bool IsPassedByReference() const { return id_ != nullptr; }

  /// Get the reference object ID.
  const ObjectID &GetReference() const {
    RAY_CHECK(id_ != nullptr) << "This argument isn't passed by reference.";
    return *id_;
  }

  /// Get the value.
  std::shared_ptr<Buffer> GetValue() const {
    RAY_CHECK(data_ != nullptr) << "This argument isn't passed by value.";
    return data_;
  }

 private:
  TaskArg(const std::shared_ptr<ObjectID> id, const std::shared_ptr<Buffer> data)
      : id_(id), data_(data) {}

  /// Id of the argument, if passed by reference, otherwise nullptr.
  const std::shared_ptr<ObjectID> id_;
  /// Data of the argument, if passed by value, otherwise nullptr.
  const std::shared_ptr<Buffer> data_;
};

enum class TaskType { NORMAL_TASK, ACTOR_CREATION_TASK, ACTOR_TASK };

/// Information of a task
struct TaskInfo {
  /// The ID of task.
  const TaskID task_id;
  /// The job ID.
  const JobID job_id;
  /// The type of task.
  const TaskType task_type;
};

/// Task specification, which includes the immutable information about the task
/// which are determined at the submission time.
/// TODO(zhijunfu): this can be removed after everything is moved to protobuf.
class TaskSpec {
 public:
  TaskSpec(const raylet::TaskSpecification &task_spec,
           const std::vector<ObjectID> &dependencies)
      : task_spec_(task_spec), dependencies_(dependencies) {}

  TaskSpec(const raylet::TaskSpecification &&task_spec,
           const std::vector<ObjectID> &&dependencies)
      : task_spec_(task_spec), dependencies_(dependencies) {}

  const raylet::TaskSpecification &GetTaskSpecification() const { return task_spec_; }

  const std::vector<ObjectID> &GetDependencies() const { return dependencies_; }

 private:
  /// Raylet task specification.
  raylet::TaskSpecification task_spec_;

  /// Dependencies.
  std::vector<ObjectID> dependencies_;
};

enum class StoreProviderType { PLASMA };

enum class TaskTransportType { RAYLET };

/// Translate from ray::rpc::Language to Language type (required by raylet client).
///
/// \param[in] language Language for a task.
/// \return Translated task language.
inline ::Language ToRayletTaskLanguage(ray::rpc::Language language) {
  switch (language) {
  case ray::rpc::Language::JAVA:
    return ::Language::JAVA;
    break;
  case ray::rpc::Language::PYTHON:
    return ::Language::PYTHON;
    break;
  case ray::rpc::Language::CPP:
    return ::Language::CPP;
    break;
  default:
    RAY_LOG(FATAL) << "Invalid language specified: " << static_cast<int>(language);
    break;
  }
}

/// Translate from Language to ray::rpc::Language type (required by core worker).
///
/// \param[in] language Language for a task.
/// \return Translated task language.
inline ray::rpc::Language ToRpcTaskLanguage(::Language language) {
  switch (language) {
  case Language::JAVA:
    return ray::rpc::Language::JAVA;
    break;
  case Language::PYTHON:
    return ray::rpc::Language::PYTHON;
    break;
  case Language::CPP:
    return ray::rpc::Language::CPP;
    break;
  default:
    RAY_LOG(FATAL) << "Invalid language specified: " << static_cast<int>(language);
    break;
  }
}

}  // namespace ray

#endif  // RAY_CORE_WORKER_COMMON_H

// Copyright 2012 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_LOGGING_LOG_H_
#define V8_LOGGING_LOG_H_

#include <atomic>
#include <memory>
#include <set>
#include <string>

#include "include/v8-callbacks.h"
#include "include/v8-profiler.h"
#include "src/base/platform/elapsed-timer.h"
#include "src/execution/isolate.h"
#include "src/logging/code-events.h"
#include "src/objects/objects.h"

namespace v8 {

namespace sampler {
class Sampler;
}  // namespace sampler

namespace internal {

struct TickSample;

// Logger is used for collecting logging information from V8 during
// execution. The result is dumped to a file.
//
// Available command line flags:
//
//  --log
// Minimal logging (no API, code, or GC sample events), default is off.
//
// --log-all
// Log all events to the file, default is off.  This is the same as combining
// --log-api and --log-code.
//
// --log-api
// Log API events to the logfile, default is off.  --log-api implies --log.
//
// --log-code
// Log code (create, move, and delete) events to the logfile, default is off.
// --log-code implies --log.
//
// --logfile <filename>
// Specify the name of the logfile, default is "v8.log".
//
// --prof
// Collect statistical profiling information (ticks), default is off.  The
// tick profiler requires code events, so --prof implies --log-code.
//
// --prof-sampling-interval <microseconds>
// The interval between --prof samples, default is 1000 microseconds (5000 on
// Android).

// Forward declarations.
class CodeEventListener;
class Isolate;
class JitLogger;
class Log;
class LowLevelLogger;
class PerfBasicLogger;
class PerfJitLogger;
class Profiler;
class SourcePosition;
class Ticker;

#undef LOG
#define LOG(isolate, Call)                                 \
  do {                                                     \
    if (v8::internal::FLAG_log) (isolate)->logger()->Call; \
  } while (false)

#define LOG_CODE_EVENT(isolate, Call)                        \
  do {                                                       \
    auto&& logger = (isolate)->logger();                     \
    if (logger->is_listening_to_code_events()) logger->Call; \
  } while (false)

class ExistingCodeLogger {
 public:
  explicit ExistingCodeLogger(Isolate* isolate,
                              CodeEventListener* listener = nullptr)
      : isolate_(isolate), listener_(listener) {}

  void LogCodeObjects();

  void LogCompiledFunctions();
  void LogExistingFunction(Handle<SharedFunctionInfo> shared,
                           Handle<AbstractCode> code,
                           CodeEventListener::LogEventsAndTags tag =
                               CodeEventListener::FUNCTION_TAG);
  void LogCodeObject(Object object);

 private:
  Isolate* isolate_;
  CodeEventListener* listener_;
};

enum class LogSeparator;

class Logger : public CodeEventListener {
 public:
  enum class ScriptEventType {
    kReserveId,
    kCreate,
    kDeserialize,
    kBackgroundCompile,
    kStreamingCompile
  };

  explicit Logger(Isolate* isolate);
  ~Logger() override;

  // The separator is used to write an unescaped "," into the log.
  static const LogSeparator kNext;

  // Acquires resources for logging if the right flags are set.
  bool SetUp(Isolate* isolate);

  // Frees resources acquired in SetUp.
  // When a temporary file is used for the log, returns its stream descriptor,
  // leaving the file open.
  V8_EXPORT_PRIVATE FILE* TearDownAndGetLogFile();

  // Sets the current code event handler.
  void SetCodeEventHandler(uint32_t options, JitCodeEventHandler event_handler);

  sampler::Sampler* sampler();
  V8_EXPORT_PRIVATE std::string file_name() const;

  V8_EXPORT_PRIVATE void StopProfilerThread();

  // Emits an event with a string value -> (name, value).
  V8_EXPORT_PRIVATE void StringEvent(const char* name, const char* value);

  // Emits an event with an int value -> (name, value).
  void IntPtrTEvent(const char* name, intptr_t value);

  // Emits an event with an handle value -> (name, location).
  void HandleEvent(const char* name, Address* location);

  // Emits memory management events for C allocated structures.
  void NewEvent(const char* name, void* object, size_t size);
  void DeleteEvent(const char* name, void* object);

  // ==== Events logged by --log-function-events ====
  void FunctionEvent(const char* reason, int script_id, double time_delta_ms,
                     int start_position, int end_position,
                     String function_name);
  void FunctionEvent(const char* reason, int script_id, double time_delta_ms,
                     int start_position, int end_position,
                     const char* function_name = nullptr,
                     size_t function_name_length = 0, bool is_one_byte = true);

  void CompilationCacheEvent(const char* action, const char* cache_type,
                             SharedFunctionInfo sfi);
  void ScriptEvent(ScriptEventType type, int script_id);
  void ScriptDetails(Script script);

  // ==== Events logged by --log-api. ====
  void ApiSecurityCheck() {
    if (!FLAG_log_api) return;
    WriteApiSecurityCheck();
  }
  void ApiNamedPropertyAccess(const char* tag, JSObject holder, Object name) {
    if (!FLAG_log_api) return;
    WriteApiNamedPropertyAccess(tag, holder, name);
  }
  void ApiIndexedPropertyAccess(const char* tag, JSObject holder,
                                uint32_t index) {
    if (!FLAG_log_api) return;
    WriteApiIndexedPropertyAccess(tag, holder, index);
  }

  void ApiObjectAccess(const char* tag, JSReceiver obj) {
    if (!FLAG_log_api) return;
    WriteApiObjectAccess(tag, obj);
  }
  void ApiEntryCall(const char* name) {
    if (!FLAG_log_api) return;
    WriteApiEntryCall(name);
  }

  // ==== Events logged by --log-code. ====
  V8_EXPORT_PRIVATE void AddCodeEventListener(CodeEventListener* listener);
  V8_EXPORT_PRIVATE void RemoveCodeEventListener(CodeEventListener* listener);

  // CodeEventListener implementation.
  void CodeCreateEvent(LogEventsAndTags tag, Handle<AbstractCode> code,
                       const char* name) override;
  void CodeCreateEvent(LogEventsAndTags tag, Handle<AbstractCode> code,
                       Handle<Name> name) override;
  void CodeCreateEvent(LogEventsAndTags tag, Handle<AbstractCode> code,
                       Handle<SharedFunctionInfo> shared,
                       Handle<Name> script_name) override;
  void CodeCreateEvent(LogEventsAndTags tag, Handle<AbstractCode> code,
                       Handle<SharedFunctionInfo> shared,
                       Handle<Name> script_name, int line, int column) override;
#if V8_ENABLE_WEBASSEMBLY
  void CodeCreateEvent(LogEventsAndTags tag, const wasm::WasmCode* code,
                       wasm::WasmName name, const char* source_url,
                       int code_offset, int script_id) override;
#endif  // V8_ENABLE_WEBASSEMBLY

  void CallbackEvent(Handle<Name> name, Address entry_point) override;
  void GetterCallbackEvent(Handle<Name> name, Address entry_point) override;
  void SetterCallbackEvent(Handle<Name> name, Address entry_point) override;
  void RegExpCodeCreateEvent(Handle<AbstractCode> code,
                             Handle<String> source) override;
  void CodeMoveEvent(AbstractCode from, AbstractCode to) override;
  void SharedFunctionInfoMoveEvent(Address from, Address to) override;
  void NativeContextMoveEvent(Address from, Address to) override {}
  void CodeMovingGCEvent() override;
  void CodeDisableOptEvent(Handle<AbstractCode> code,
                           Handle<SharedFunctionInfo> shared) override;
  void CodeDeoptEvent(Handle<Code> code, DeoptimizeKind kind, Address pc,
                      int fp_to_sp_delta) override;
  void CodeDependencyChangeEvent(Handle<Code> code,
                                 Handle<SharedFunctionInfo> sfi,
                                 const char* reason) override;
  void FeedbackVectorEvent(FeedbackVector vector, AbstractCode code);
  void WeakCodeClearEvent() override {}

  void ProcessDeoptEvent(Handle<Code> code, SourcePosition position,
                         const char* kind, const char* reason);

  // Emits a code line info record event.
  void CodeLinePosInfoRecordEvent(Address code_start,
                                  ByteArray source_position_table,
                                  JitCodeEvent::CodeType code_type);
#if V8_ENABLE_WEBASSEMBLY
  void WasmCodeLinePosInfoRecordEvent(
      Address code_start, base::Vector<const byte> source_position_table);
#endif  // V8_ENABLE_WEBASSEMBLY

  void CodeNameEvent(Address addr, int pos, const char* code_name);

  void ICEvent(const char* type, bool keyed, Handle<Map> map,
               Handle<Object> key, char old_state, char new_state,
               const char* modifier, const char* slow_stub_reason);

  void MapEvent(const char* type, Handle<Map> from, Handle<Map> to,
                const char* reason = nullptr,
                Handle<HeapObject> name_or_sfi = Handle<HeapObject>());
  void MapCreate(Map map);
  void MapDetails(Map map);

  void SharedLibraryEvent(const std::string& library_path, uintptr_t start,
                          uintptr_t end, intptr_t aslr_slide);
  void SharedLibraryEnd();

  void CurrentTimeEvent();

  V8_EXPORT_PRIVATE void TimerEvent(v8::LogEventStatus se, const char* name);

  void BasicBlockCounterEvent(const char* name, int block_id, uint32_t count);

  void BuiltinHashEvent(const char* name, int hash);

  static void EnterExternal(Isolate* isolate);
  static void LeaveExternal(Isolate* isolate);

  static void DefaultEventLoggerSentinel(const char* name, int event) {}

  V8_INLINE static void CallEventLoggerInternal(Isolate* isolate,
                                                const char* name,
                                                v8::LogEventStatus se,
                                                bool expose_to_api) {
    if (isolate->event_logger() == DefaultEventLoggerSentinel) {
      LOG(isolate, TimerEvent(se, name));
    } else if (expose_to_api) {
      isolate->event_logger()(name, static_cast<v8::LogEventStatus>(se));
    }
  }

  V8_INLINE static void CallEventLogger(Isolate* isolate, const char* name,
                                        v8::LogEventStatus se,
                                        bool expose_to_api) {
    if (!isolate->event_logger()) return;
    CallEventLoggerInternal(isolate, name, se, expose_to_api);
  }

  V8_EXPORT_PRIVATE bool is_logging();

  bool is_listening_to_code_events() override {
    return is_logging() || jit_logger_ != nullptr;
  }

  void LogExistingFunction(Handle<SharedFunctionInfo> shared,
                           Handle<AbstractCode> code);
  // Logs all compiled functions found in the heap.
  V8_EXPORT_PRIVATE void LogCompiledFunctions();
  // Logs all accessor callbacks found in the heap.
  V8_EXPORT_PRIVATE void LogAccessorCallbacks();
  // Used for logging stubs found in the snapshot.
  V8_EXPORT_PRIVATE void LogCodeObjects();
  // Logs all Maps found on the heap.
  void LogAllMaps();

  // Converts tag to a corresponding NATIVE_... if the script is native.
  V8_INLINE static CodeEventListener::LogEventsAndTags ToNativeByScript(
      CodeEventListener::LogEventsAndTags, Script);

 private:
  void UpdateIsLogging(bool value);

  // Emits the profiler's first message.
  void ProfilerBeginEvent();

  // Emits callback event messages.
  void CallbackEventInternal(const char* prefix, Handle<Name> name,
                             Address entry_point);

  // Internal configurable move event.
  void MoveEventInternal(CodeEventListener::LogEventsAndTags event,
                         Address from, Address to);

  // Helper method. It resets name_buffer_ and add tag name into it.
  void InitNameBuffer(CodeEventListener::LogEventsAndTags tag);

  // Emits a profiler tick event. Used by the profiler thread.
  void TickEvent(TickSample* sample, bool overflow);
  void RuntimeCallTimerEvent();

  // Logs a StringEvent regardless of whether FLAG_log is true.
  void UncheckedStringEvent(const char* name, const char* value);

  // Logs a scripts sources. Keeps track of all logged scripts to ensure that
  // each script is logged only once.
  bool EnsureLogScriptSource(Script script);

  void LogSourceCodeInformation(Handle<AbstractCode> code,
                                Handle<SharedFunctionInfo> shared);
  void LogCodeDisassemble(Handle<AbstractCode> code);

  void WriteApiSecurityCheck();
  void WriteApiNamedPropertyAccess(const char* tag, JSObject holder,
                                   Object name);
  void WriteApiIndexedPropertyAccess(const char* tag, JSObject holder,
                                     uint32_t index);
  void WriteApiObjectAccess(const char* tag, JSReceiver obj);
  void WriteApiEntryCall(const char* name);

  int64_t Time();

  Isolate* isolate_;

  // The sampler used by the profiler and the sliding state window.
  std::unique_ptr<Ticker> ticker_;

  // When the statistical profile is active, profiler_
  // points to a Profiler, that handles collection
  // of samples.
  std::unique_ptr<Profiler> profiler_;

  // Internal implementation classes with access to private members.
  friend class Profiler;

  std::atomic<bool> is_logging_;
  std::unique_ptr<Log> log_;
#if V8_OS_LINUX
  std::unique_ptr<PerfBasicLogger> perf_basic_logger_;
  std::unique_ptr<PerfJitLogger> perf_jit_logger_;
#endif
  std::unique_ptr<LowLevelLogger> ll_logger_;
  std::unique_ptr<JitLogger> jit_logger_;
  std::set<int> logged_source_code_;
  uint32_t next_source_info_id_ = 0;

  // Guards against multiple calls to TearDown() that can happen in some tests.
  // 'true' between SetUp() and TearDown().
  bool is_initialized_;

  ExistingCodeLogger existing_code_logger_;

  base::ElapsedTimer timer_;
};

#define TIMER_EVENTS_LIST(V)     \
  V(RecompileSynchronous, true)  \
  V(RecompileConcurrent, true)   \
  V(CompileIgnition, true)       \
  V(CompileFullCode, true)       \
  V(OptimizeCode, true)          \
  V(CompileCode, true)           \
  V(CompileCodeBackground, true) \
  V(DeoptimizeCode, true)        \
  V(Execute, true)

#define V(TimerName, expose)                          \
  class TimerEvent##TimerName : public AllStatic {    \
   public:                                            \
    static const char* name(void* unused = nullptr) { \
      return "V8." #TimerName;                        \
    }                                                 \
    static bool expose_to_api() { return expose; }    \
  };
TIMER_EVENTS_LIST(V)
#undef V

template <class TimerEvent>
class V8_NODISCARD TimerEventScope {
 public:
  explicit TimerEventScope(Isolate* isolate) : isolate_(isolate) {
    LogTimerEvent(v8::LogEventStatus::kStart);
  }

  ~TimerEventScope() { LogTimerEvent(v8::LogEventStatus::kEnd); }

 private:
  void LogTimerEvent(v8::LogEventStatus se);
  Isolate* isolate_;
};

// Abstract
class V8_EXPORT_PRIVATE CodeEventLogger : public CodeEventListener {
 public:
  explicit CodeEventLogger(Isolate* isolate);
  ~CodeEventLogger() override;

  void CodeCreateEvent(LogEventsAndTags tag, Handle<AbstractCode> code,
                       const char* name) override;
  void CodeCreateEvent(LogEventsAndTags tag, Handle<AbstractCode> code,
                       Handle<Name> name) override;
  void CodeCreateEvent(LogEventsAndTags tag, Handle<AbstractCode> code,
                       Handle<SharedFunctionInfo> shared,
                       Handle<Name> script_name) override;
  void CodeCreateEvent(LogEventsAndTags tag, Handle<AbstractCode> code,
                       Handle<SharedFunctionInfo> shared,
                       Handle<Name> script_name, int line, int column) override;
#if V8_ENABLE_WEBASSEMBLY
  void CodeCreateEvent(LogEventsAndTags tag, const wasm::WasmCode* code,
                       wasm::WasmName name, const char* source_url,
                       int code_offset, int script_id) override;
#endif  // V8_ENABLE_WEBASSEMBLY

  void RegExpCodeCreateEvent(Handle<AbstractCode> code,
                             Handle<String> source) override;
  void CallbackEvent(Handle<Name> name, Address entry_point) override {}
  void GetterCallbackEvent(Handle<Name> name, Address entry_point) override {}
  void SetterCallbackEvent(Handle<Name> name, Address entry_point) override {}
  void SharedFunctionInfoMoveEvent(Address from, Address to) override {}
  void NativeContextMoveEvent(Address from, Address to) override {}
  void CodeMovingGCEvent() override {}
  void CodeDeoptEvent(Handle<Code> code, DeoptimizeKind kind, Address pc,
                      int fp_to_sp_delta) override {}
  void CodeDependencyChangeEvent(Handle<Code> code,
                                 Handle<SharedFunctionInfo> sfi,
                                 const char* reason) override {}
  void WeakCodeClearEvent() override {}

 protected:
  Isolate* isolate_;

 private:
  class NameBuffer;

  virtual void LogRecordedBuffer(Handle<AbstractCode> code,
                                 MaybeHandle<SharedFunctionInfo> maybe_shared,
                                 const char* name, int length) = 0;
#if V8_ENABLE_WEBASSEMBLY
  virtual void LogRecordedBuffer(const wasm::WasmCode* code, const char* name,
                                 int length) = 0;
#endif  // V8_ENABLE_WEBASSEMBLY

  std::unique_ptr<NameBuffer> name_buffer_;
};

struct CodeEvent {
  Isolate* isolate_;
  uintptr_t code_start_address;
  size_t code_size;
  Handle<String> function_name;
  Handle<String> script_name;
  int script_line;
  int script_column;
  CodeEventType code_type;
  const char* comment;
  uintptr_t previous_code_start_address;
};

class ExternalCodeEventListener : public CodeEventListener {
 public:
  explicit ExternalCodeEventListener(Isolate* isolate);
  ~ExternalCodeEventListener() override;

  void CodeCreateEvent(LogEventsAndTags tag, Handle<AbstractCode> code,
                       const char* comment) override;
  void CodeCreateEvent(LogEventsAndTags tag, Handle<AbstractCode> code,
                       Handle<Name> name) override;
  void CodeCreateEvent(LogEventsAndTags tag, Handle<AbstractCode> code,
                       Handle<SharedFunctionInfo> shared,
                       Handle<Name> name) override;
  void CodeCreateEvent(LogEventsAndTags tag, Handle<AbstractCode> code,
                       Handle<SharedFunctionInfo> shared, Handle<Name> source,
                       int line, int column) override;
#if V8_ENABLE_WEBASSEMBLY
  void CodeCreateEvent(LogEventsAndTags tag, const wasm::WasmCode* code,
                       wasm::WasmName name, const char* source_url,
                       int code_offset, int script_id) override;
#endif  // V8_ENABLE_WEBASSEMBLY

  void RegExpCodeCreateEvent(Handle<AbstractCode> code,
                             Handle<String> source) override;
  void CallbackEvent(Handle<Name> name, Address entry_point) override {}
  void GetterCallbackEvent(Handle<Name> name, Address entry_point) override {}
  void SetterCallbackEvent(Handle<Name> name, Address entry_point) override {}
  void SharedFunctionInfoMoveEvent(Address from, Address to) override {}
  void NativeContextMoveEvent(Address from, Address to) override {}
  void CodeMoveEvent(AbstractCode from, AbstractCode to) override;
  void CodeDisableOptEvent(Handle<AbstractCode> code,
                           Handle<SharedFunctionInfo> shared) override {}
  void CodeMovingGCEvent() override {}
  void CodeDeoptEvent(Handle<Code> code, DeoptimizeKind kind, Address pc,
                      int fp_to_sp_delta) override {}
  void CodeDependencyChangeEvent(Handle<Code> code,
                                 Handle<SharedFunctionInfo> sfi,
                                 const char* reason) override {}
  void WeakCodeClearEvent() override {}

  void StartListening(v8::CodeEventHandler* code_event_handler);
  void StopListening();

  bool is_listening_to_code_events() override { return true; }

 private:
  void LogExistingCode();

  bool is_listening_;
  Isolate* isolate_;
  v8::CodeEventHandler* code_event_handler_;
};

}  // namespace internal
}  // namespace v8

#endif  // V8_LOGGING_LOG_H_

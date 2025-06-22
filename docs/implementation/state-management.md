# 상태 관리 및 메모리

## WorkingMemory

```python
class WorkingMemory:
    """에이전트 작업 메모리"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.memory = {}
        self.access_count = {}
        self.timestamps = {}
    
    def store(self, key: str, value: Any, ttl: Optional[int] = None):
        """메모리에 저장"""
        if len(self.memory) >= self.max_size:
            self._evict_lru()
        
        self.memory[key] = value
        self.access_count[key] = 0
        self.timestamps[key] = time.time()
        
        if ttl:
            asyncio.create_task(self._schedule_cleanup(key, ttl))
    
    def retrieve(self, key: str) -> Optional[Any]:
        """메모리에서 조회"""
        if key in self.memory:
            self.access_count[key] += 1
            return self.memory[key]
        return None
    
    def remove(self, key: str):
        """메모리에서 제거"""
        if key in self.memory:
            del self.memory[key]
            del self.access_count[key]
            del self.timestamps[key]
    
    def _evict_lru(self):
        """LRU 정책으로 메모리 정리"""
        if not self.memory:
            return
        
        # 가장 적게 사용된 항목 제거
        lru_key = min(self.access_count.items(), key=lambda x: x[1])[0]
        self.remove(lru_key)
    
    async def _schedule_cleanup(self, key: str, ttl: int):
        """TTL 기반 자동 정리"""
        await asyncio.sleep(ttl)
        self.remove(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """메모리 사용 통계"""
        return {
            "total_items": len(self.memory),
            "max_size": self.max_size,
            "utilization": len(self.memory) / self.max_size,
            "most_accessed": max(self.access_count.items(), key=lambda x: x[1])[0] if self.access_count else None,
            "oldest_item": min(self.timestamps.items(), key=lambda x: x[1])[0] if self.timestamps else None
        }
```

## SessionState

```python
class SessionState:
    """평가 세션 상태 관리"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = datetime.now()
        self.current_phase = None
        self.phase_states = {}
        self.global_state = {}
        self.checkpoints = []
    
    def save_checkpoint(self, phase: str, state: Dict[str, Any]):
        """체크포인트 저장"""
        checkpoint = {
            "phase": phase,
            "timestamp": datetime.now(),
            "state": state,
            "checkpoint_id": f"{phase}_{len(self.checkpoints)}"
        }
        self.checkpoints.append(checkpoint)
    
    def restore_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """체크포인트 복원"""
        for checkpoint in self.checkpoints:
            if checkpoint["checkpoint_id"] == checkpoint_id:
                return checkpoint["state"]
        return None
    
    def update_phase_state(self, phase: str, key: str, value: Any):
        """Phase별 상태 업데이트"""
        if phase not in self.phase_states:
            self.phase_states[phase] = {}
        self.phase_states[phase][key] = value
    
    def get_phase_state(self, phase: str, key: str = None) -> Any:
        """Phase별 상태 조회"""
        if phase not in self.phase_states:
            return None
        if key is None:
            return self.phase_states[phase]
        return self.phase_states[phase].get(key)
    
    def persist_to_disk(self, file_path: str):
        """디스크에 상태 저장"""
        state_data = {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "current_phase": self.current_phase,
            "phase_states": self.phase_states,
            "global_state": self.global_state,
            "checkpoints": [
                {
                    **cp,
                    "timestamp": cp["timestamp"].isoformat()
                }
                for cp in self.checkpoints
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_disk(cls, file_path: str) -> 'SessionState':
        """디스크에서 상태 복원"""
        with open(file_path, 'r') as f:
            state_data = json.load(f)
        
        session = cls(state_data["session_id"])
        session.start_time = datetime.fromisoformat(state_data["start_time"])
        session.current_phase = state_data["current_phase"]
        session.phase_states = state_data["phase_states"]
        session.global_state = state_data["global_state"]
        
        # 체크포인트 복원
        for cp_data in state_data["checkpoints"]:
            checkpoint = {
                **cp_data,
                "timestamp": datetime.fromisoformat(cp_data["timestamp"])
            }
            session.checkpoints.append(checkpoint)
        
        return session
    
    def get_session_summary(self) -> Dict[str, Any]:
        """세션 요약 정보"""
        duration = datetime.now() - self.start_time
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "current_phase": self.current_phase,
            "completed_phases": list(self.phase_states.keys()),
            "checkpoint_count": len(self.checkpoints)
        }
```

## 에이전트 안전성 및 제약

### Resource Management

```python
class ResourceManager:
    """시스템 리소스 관리 및 제한"""
    
    def __init__(self, config: ResourceConfig):
        self.max_memory_mb = config.max_memory_mb
        self.max_cpu_percent = config.max_cpu_percent
        self.max_disk_gb = config.max_disk_gb
        self.max_execution_time = config.max_execution_time
        
        self.current_usage = ResourceUsage()
        self._monitoring_task = None
    
    async def start_monitoring(self):
        """리소스 모니터링 시작"""
        self._monitoring_task = asyncio.create_task(self._monitor_resources())
    
    async def stop_monitoring(self):
        """리소스 모니터링 중단"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
    
    async def _monitor_resources(self):
        """주기적 리소스 사용량 체크"""
        while True:
            try:
                usage = await self._get_current_usage()
                
                if usage.memory_mb > self.max_memory_mb:
                    await self._handle_memory_limit()
                
                if usage.cpu_percent > self.max_cpu_percent:
                    await self._handle_cpu_limit()
                
                if usage.disk_gb > self.max_disk_gb:
                    await self._handle_disk_limit()
                
                await asyncio.sleep(5)  # 5초마다 체크
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
    
    async def _get_current_usage(self) -> ResourceUsage:
        """현재 리소스 사용량 조회"""
        import psutil
        
        memory_mb = psutil.virtual_memory().used / (1024 * 1024)
        cpu_percent = psutil.cpu_percent(interval=1)
        disk_gb = psutil.disk_usage('/').used / (1024 * 1024 * 1024)
        
        return ResourceUsage(
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            disk_gb=disk_gb
        )
    
    async def _handle_memory_limit(self):
        """메모리 한계 처리"""
        # 캐시 정리
        await self._clear_caches()
        
        # 가비지 컬렉션 강제 실행
        import gc
        gc.collect()
        
        # 그래도 한계 초과시 예외 발생
        current_usage = await self._get_current_usage()
        if current_usage.memory_mb > self.max_memory_mb:
            raise ResourceLimitExceeded("Memory limit exceeded")
    
    async def _handle_cpu_limit(self):
        """CPU 한계 처리"""
        # 실행 우선순위 조정
        await asyncio.sleep(1)  # 잠시 대기
    
    async def _handle_disk_limit(self):
        """디스크 한계 처리"""
        # 임시 파일 정리
        await self._cleanup_temp_files()
        
        current_usage = await self._get_current_usage()
        if current_usage.disk_gb > self.max_disk_gb:
            raise ResourceLimitExceeded("Disk limit exceeded")

@dataclass
class ResourceUsage:
    memory_mb: float
    cpu_percent: float
    disk_gb: float

@dataclass
class ResourceConfig:
    max_memory_mb: int = 4096  # 4GB
    max_cpu_percent: float = 80.0
    max_disk_gb: float = 10.0
    max_execution_time: int = 3600  # 1시간

class ResourceLimitExceeded(Exception):
    """리소스 한계 초과 예외"""
    pass
```

### Security Constraints

```python
class SecurityManager:
    """보안 제약 및 접근 제어"""
    
    def __init__(self, config: SecurityConfig):
        self.allowed_paths = config.allowed_paths
        self.forbidden_commands = config.forbidden_commands
        self.audit_log = AuditLog()
    
    def validate_file_access(self, file_path: str, operation: str) -> bool:
        """파일 접근 권한 검증"""
        abs_path = os.path.abspath(file_path)
        
        # 허용된 경로 내부인지 확인
        for allowed_path in self.allowed_paths:
            if abs_path.startswith(os.path.abspath(allowed_path)):
                self.audit_log.log_access(abs_path, operation, "ALLOWED")
                return True
        
        self.audit_log.log_access(abs_path, operation, "DENIED")
        return False
    
    def validate_command(self, command: List[str]) -> bool:
        """명령어 실행 권한 검증"""
        cmd_name = command[0] if command else ""
        
        if cmd_name in self.forbidden_commands:
            self.audit_log.log_command(command, "DENIED")
            return False
        
        # 특별 제약: selvage-deprecated는 읽기 전용
        if "selvage-deprecated" in " ".join(command):
            if any(write_op in " ".join(command) 
                   for write_op in ["commit", "push", "rm", "mv"]):
                self.audit_log.log_command(command, "DENIED - READ_ONLY")
                return False
        
        self.audit_log.log_command(command, "ALLOWED") 
        return True

@dataclass
class SecurityConfig:
    allowed_paths: List[str]
    forbidden_commands: List[str]

class AuditLog:
    """보안 감사 로그"""
    
    def __init__(self):
        self.logs = []
    
    def log_access(self, path: str, operation: str, result: str):
        """파일 접근 로그"""
        self.logs.append({
            "type": "file_access",
            "timestamp": datetime.now().isoformat(),
            "path": path,
            "operation": operation,
            "result": result
        })
    
    def log_command(self, command: List[str], result: str):
        """명령어 실행 로그"""
        self.logs.append({
            "type": "command_execution",
            "timestamp": datetime.now().isoformat(),
            "command": " ".join(command),
            "result": result
        })
    
    def export_logs(self, file_path: str):
        """로그 내보내기"""
        with open(file_path, 'w') as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False)
```
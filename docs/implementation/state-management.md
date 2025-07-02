# 상태 관리

## SessionState

```python
class SessionState:
    """평가 세션 상태 관리"""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or self._generate_session_id()
        self.start_time = datetime.now()
        self.current_phase = None
        self.phase_states = {}
        self.global_state = {}
    
    def set_current_phase(self, phase: str):
        """현재 실행 중인 Phase 설정"""
        self.current_phase = phase
        if phase not in self.phase_states:
            self.phase_states[phase] = {}
    
    def update_phase_state(self, phase: str, state_updates: Dict[str, Any]):
        """특정 Phase의 상태 업데이트"""
        if phase not in self.phase_states:
            self.phase_states[phase] = {}
        self.phase_states[phase].update(state_updates)
    
    def get_phase_state(self, phase: str) -> Dict[str, Any]:
        """특정 Phase의 상태 조회"""
        return self.phase_states.get(phase, {})
    
    def update_global_state(self, state_updates: Dict[str, Any]):
        """전역 상태 업데이트"""
        self.global_state.update(state_updates)
    
    def mark_phase_completed(self, phase: str):
        """Phase 완료 표시"""
        self.update_phase_state(phase, {
            "completed": True, 
            "completed_at": datetime.now().isoformat()
        })
    
    def is_phase_completed(self, phase: str) -> bool:
        """Phase 완료 여부 확인"""
        return self.phase_states.get(phase, {}).get("completed", False)
    
    def get_completed_phases(self) -> List[str]:
        """완료된 Phase 목록 반환"""
        completed = []
        for phase in ["commit_collection", "review_execution", "deepeval_conversion", "analysis"]:
            if self.is_phase_completed(phase):
                completed.append(phase)
        return completed
    
    async def persist_to_disk(self, file_path: str):
        """디스크에 상태 저장"""
        state_data = {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "current_phase": self.current_phase,
            "phase_states": self.phase_states,
            "global_state": self.global_state
        }
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    async def load_from_disk(cls, file_path: str) -> 'SessionState':
        """디스크에서 상태 복원"""
        with open(file_path, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
        
        session = cls(session_id=state_data["session_id"])
        session.start_time = datetime.fromisoformat(state_data["start_time"])
        session.current_phase = state_data["current_phase"]
        session.phase_states = state_data["phase_states"]
        session.global_state = state_data["global_state"]
        
        return session
    
    async def auto_persist(self, output_dir: str, interval: int = 300):
        """자동 영속화 시작 (5분마다)"""
        state_file = os.path.join(output_dir, "session_state.json")
        
        async def persist_loop():
            while True:
                try:
                    await asyncio.sleep(interval)
                    await self.persist_to_disk(state_file)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Auto-persist failed: {e}")
        
        task = asyncio.create_task(persist_loop())
        return task
    
    def get_session_summary(self) -> Dict[str, Any]:
        """세션 요약 정보"""
        duration = datetime.now() - self.start_time
        completed_phases = self.get_completed_phases()
        
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "current_phase": self.current_phase,
            "completed_phases": completed_phases
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
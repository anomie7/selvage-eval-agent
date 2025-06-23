"""작업 메모리 관리

에이전트가 실행 중에 사용하는 작업 메모리를 관리합니다.
LRU 정책과 TTL 기반 자동 정리 기능을 제공합니다.
"""

import asyncio
import time
from typing import Any, Optional, Dict
import gc
import logging

logger = logging.getLogger(__name__)


class WorkingMemory:
    """에이전트 작업 메모리"""
    
    def __init__(self, max_size: int = 1000):
        """작업 메모리 초기화
        
        Args:
            max_size: 최대 저장 가능한 항목 수
        """
        self.max_size = max_size
        self.memory: Dict[str, Any] = {}
        self.access_count: Dict[str, int] = {}
        self.timestamps: Dict[str, float] = {}
        self._cleanup_tasks: Dict[str, asyncio.Task] = {}
    
    def store(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """메모리에 데이터 저장
        
        Args:
            key: 저장할 데이터의 키
            value: 저장할 데이터
            ttl: Time To Live (초), None이면 영구 저장
        """
        if len(self.memory) >= self.max_size:
            self._evict_lru()
        
        # 기존 TTL 작업 취소
        if key in self._cleanup_tasks:
            self._cleanup_tasks[key].cancel()
            del self._cleanup_tasks[key]
        
        self.memory[key] = value
        self.access_count[key] = 0
        self.timestamps[key] = time.time()
        
        if ttl:
            task = asyncio.create_task(self._schedule_cleanup(key, ttl))
            self._cleanup_tasks[key] = task
        
        logger.debug(f"Stored item in memory: {key} (TTL: {ttl})")
    
    def retrieve(self, key: str) -> Optional[Any]:
        """메모리에서 데이터 조회
        
        Args:
            key: 조회할 데이터의 키
            
        Returns:
            저장된 데이터 또는 None
        """
        if key in self.memory:
            self.access_count[key] += 1
            logger.debug(f"Retrieved item from memory: {key} (access count: {self.access_count[key]})")
            return self.memory[key]
        return None
    
    def remove(self, key: str) -> bool:
        """메모리에서 데이터 제거
        
        Args:
            key: 제거할 데이터의 키
            
        Returns:
            제거 성공 여부
        """
        if key in self.memory:
            del self.memory[key]
            del self.access_count[key]
            del self.timestamps[key]
            
            # TTL 작업 취소
            if key in self._cleanup_tasks:
                self._cleanup_tasks[key].cancel()
                del self._cleanup_tasks[key]
            
            logger.debug(f"Removed item from memory: {key}")
            return True
        return False
    
    def clear(self) -> None:
        """메모리 전체 정리"""
        # 모든 TTL 작업 취소
        for task in self._cleanup_tasks.values():
            task.cancel()
        
        self.memory.clear()
        self.access_count.clear()
        self.timestamps.clear()
        self._cleanup_tasks.clear()
        
        logger.info("Cleared all memory")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 사용 통계 반환
        
        Returns:
            메모리 사용 통계 딕셔너리
        """
        current_time = time.time()
        
        return {
            "total_items": len(self.memory),
            "max_capacity": self.max_size,
            "usage_percent": (len(self.memory) / self.max_size) * 100,
            "oldest_item_age": current_time - min(self.timestamps.values()) if self.timestamps else 0,
            "most_accessed_key": max(self.access_count.items(), key=lambda x: x[1])[0] if self.access_count else None,
            "active_ttl_tasks": len(self._cleanup_tasks)
        }
    
    def _evict_lru(self) -> None:
        """LRU 정책으로 메모리 정리"""
        if not self.memory:
            return
        
        # 가장 적게 사용된 항목 제거
        lru_key = min(self.access_count.items(), key=lambda x: x[1])[0]
        self.remove(lru_key)
        logger.debug(f"Evicted LRU item: {lru_key}")
    
    async def _schedule_cleanup(self, key: str, ttl: int) -> None:
        """TTL 기반 자동 정리
        
        Args:
            key: 정리할 데이터의 키
            ttl: Time To Live (초)
        """
        try:
            await asyncio.sleep(ttl)
            if key in self.memory:
                self.remove(key)
                logger.debug(f"TTL expired, removed item: {key}")
        except asyncio.CancelledError:
            logger.debug(f"TTL cleanup cancelled for item: {key}")
    
    def force_gc(self) -> None:
        """가비지 컬렉션 강제 실행"""
        collected = gc.collect()
        logger.debug(f"Forced garbage collection, collected {collected} objects")


class MemoryCache:
    """특정 용도별 캐시 관리"""
    
    def __init__(self, working_memory: WorkingMemory, prefix: str):
        """캐시 초기화
        
        Args:
            working_memory: 사용할 작업 메모리 인스턴스
            prefix: 캐시 키 접두사
        """
        self.working_memory = working_memory
        self.prefix = prefix
    
    def _make_key(self, key: str) -> str:
        """캐시 키 생성"""
        return f"{self.prefix}:{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """캐시에서 데이터 조회"""
        return self.working_memory.retrieve(self._make_key(key))
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """캐시에 데이터 저장"""
        self.working_memory.store(self._make_key(key), value, ttl)
    
    async def delete(self, key: str) -> bool:
        """캐시에서 데이터 삭제"""
        return self.working_memory.remove(self._make_key(key))
    
    async def exists(self, key: str) -> bool:
        """캐시에 데이터 존재 여부 확인"""
        return self.working_memory.retrieve(self._make_key(key)) is not None


# 전역 작업 메모리 인스턴스 (싱글톤 패턴)
_global_working_memory: Optional[WorkingMemory] = None


def get_working_memory() -> WorkingMemory:
    """전역 작업 메모리 인스턴스 반환"""
    global _global_working_memory
    if _global_working_memory is None:
        _global_working_memory = WorkingMemory()
    return _global_working_memory


def create_cache(prefix: str) -> MemoryCache:
    """특정 용도별 캐시 생성
    
    Args:
        prefix: 캐시 키 접두사
        
    Returns:
        생성된 캐시 인스턴스
    """
    return MemoryCache(get_working_memory(), prefix)
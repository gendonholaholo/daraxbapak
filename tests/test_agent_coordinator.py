import pytest
from src.agents.coordinator import AgentCoordinator, AgentTask, AgentRole
from unittest.mock import patch, MagicMock
import asyncio

@pytest.fixture
def coordinator():
    return AgentCoordinator()

@pytest.fixture
def sample_tasks():
    return [
        AgentTask(
            role=AgentRole.ANALYZER,
            input_data={"text": "Analyze this text"},
            priority=1
        ),
        AgentTask(
            role=AgentRole.RESEARCHER,
            input_data={"query": "Research this topic"},
            priority=2
        )
    ]

@pytest.mark.asyncio
async def test_task_submission(coordinator, sample_tasks):
    # Submit tasks
    task_ids = []
    for task in sample_tasks:
        task_id = await coordinator.submit_task(task)
        task_ids.append(task_id)
    
    # Verify task IDs
    assert len(task_ids) == 2
    assert all(isinstance(task_id, str) for task_id in task_ids)
    assert all(task_id.startswith(role.value) for task_id, role in zip(task_ids, [AgentRole.ANALYZER, AgentRole.RESEARCHER]))

@pytest.mark.asyncio
async def test_task_processing(coordinator, sample_tasks):
    with patch("src.providers.base.BaseProvider.chat_completion") as mock_completion:
        # Setup mock completion response
        mock_completion.return_value = MagicMock(
            content="Test response",
            usage={"total_tokens": 100},
            model="gpt-3.5-turbo",
            provider="openai"
        )
        
        # Submit and process tasks
        for task in sample_tasks:
            await coordinator.submit_task(task)
        
        results = await coordinator.process_tasks()
        
        # Verify results
        assert len(results) == 2
        for task_id, result in results.items():
            assert result["status"] == "completed"
            assert result["result"] == "Test response"
            assert "role" in result

@pytest.mark.asyncio
async def test_task_dependencies(coordinator):
    # Create tasks with dependencies
    task1 = AgentTask(
        role=AgentRole.ANALYZER,
        input_data={"text": "First task"},
        priority=1
    )
    
    task2 = AgentTask(
        role=AgentRole.RESEARCHER,
        input_data={"query": "Second task"},
        dependencies=["analyzer_0"],  # Depends on first task
        priority=2
    )
    
    with patch("src.providers.base.BaseProvider.chat_completion") as mock_completion:
        # Setup mock completion response
        mock_completion.return_value = MagicMock(
            content="Test response",
            usage={"total_tokens": 100},
            model="gpt-3.5-turbo",
            provider="openai"
        )
        
        # Submit tasks
        await coordinator.submit_task(task1)
        await coordinator.submit_task(task2)
        
        # Process tasks
        results = await coordinator.process_tasks()
        
        # Verify both tasks completed
        assert len(results) == 2
        assert all(result["status"] == "completed" for result in results.values())

@pytest.mark.asyncio
async def test_task_timeout(coordinator, sample_tasks):
    with patch("src.providers.base.BaseProvider.chat_completion") as mock_completion:
        # Setup mock to simulate long-running task
        async def slow_completion(*args, **kwargs):
            await asyncio.sleep(2)  # Simulate delay
            return MagicMock(
                content="Test response",
                usage={"total_tokens": 100},
                model="gpt-3.5-turbo",
                provider="openai"
            )
        
        mock_completion.side_effect = slow_completion
        
        # Submit tasks
        for task in sample_tasks:
            await coordinator.submit_task(task)
        
        # Process tasks with short timeout
        results = await coordinator.coordinate(sample_tasks, timeout=0.5)
        
        # Verify timeout occurred
        assert results["status"] == "timeout"
        assert "completed_tasks" in results 
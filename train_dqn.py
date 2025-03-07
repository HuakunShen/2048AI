import torch
from src.game.controller.game import Game
from src.agent.agentImpl import DQNPlayer
from src.game.model.staticboardImpl import TorchStaticBoard

def get_device():
    """Get the best available device (MPS for Apple Silicon, CUDA for NVIDIA, or CPU)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    # Set up device
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize game with TorchStaticBoard for tensor operations
    game = Game(static_board=TorchStaticBoard)
    
    # Create DQN agent with device specification
    agent = DQNPlayer(
        game=game,
        quiet=False,
        ui=True,
        memory_size=50000,
        batch_size=128,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        learning_rate=0.001,
        device=device  # Pass device to agent
    )
    
    # Train the agent
    print("Starting training...")
    agent.train(episodes=1000, target_update=10)
    
    # Save the trained model
    agent.save_model("models/dqn_2048.pth")
    
    # Test the trained agent
    print("\nTesting trained agent...")
    game.restart()
    score, max_val, runtime = agent.run()
    print(f"Test Results - Score: {score}, Max Value: {max_val}, Runtime: {runtime:.2f}s")

if __name__ == "__main__":
    main()

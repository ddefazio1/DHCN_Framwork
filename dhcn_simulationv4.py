import numpy as np
import matplotlib.pyplot as plt
import csv

# -------------------------------
# Simulation Parameters
# -------------------------------
num_agents = 10
num_steps = 60
phase_duration = 15
ai_base_influence = 0.1
noise_level = 0.05
collapse_threshold = 0.02
memory_influence = 0.02

# -------------------------------
# Initialize agents
# -------------------------------
agents = np.random.rand(num_agents)
history = np.zeros((num_steps, num_agents))
phase_history = np.zeros(num_steps)
collapse_events = []
ai_memory_history = np.zeros(num_steps)

ai_memory = 0.0

# -------------------------------
# Simulation loop
# -------------------------------
for t in range(num_steps):
    phase = min(t // phase_duration + 1, 4)
    phase_history[t] = phase
    
    # Shared insight with memory
    shared_insight = np.mean(agents)
    if phase == 4:
        shared_insight += ai_memory

    # Update agents
    for i in range(num_agents):
        influence = ai_base_influence * (2 ** (phase - 1))
        agents[i] += influence * (shared_insight - agents[i]) + noise_level * np.random.randn()
        agents[i] = np.clip(agents[i], 0, 1)
    
    history[t] = agents

    # Check collective decision
    variance = np.var(agents)
    if variance < collapse_threshold and (len(collapse_events) == 0 or t > collapse_events[-1] + 1):
        collapse_events.append(t)
        if phase == 4:
            ai_memory += memory_influence

    ai_memory_history[t] = ai_memory

# -------------------------------
# Visualization
# -------------------------------
plt.figure(figsize=(14,6))
phase_colors = {1:'#a6cee3', 2:'#1f78b4', 3:'#b2df8a', 4:'#33a02c'}

# Agent states
for i in range(num_agents):
    plt.plot(history[:,i], label=f'Agent {i+1}', lw=1.2)

# Phase shading
for phase in range(1,5):
    start = (phase-1)*phase_duration
    end = min(phase*phase_duration, num_steps)
    plt.axvspan(start, end, color=phase_colors[phase], alpha=0.1, label=f'Phase {phase}' if i==0 else "")

# Collapse events
for t in collapse_events:
    plt.axvline(t, color='red', linestyle='--', lw=1.5, label='Collective Decision' if t==collapse_events[0] else "")

# AI memory line
plt.plot(ai_memory_history, color='black', lw=2.5, label='AI Memory (Phase 4)')

# AI memory overlay
plt.fill_between(range(num_steps), 0, ai_memory_history, color='gray', alpha=0.15, label='Cumulative Supermind Strength')

plt.xlabel('Time Step')
plt.ylabel('Agent State / AI Memory')
plt.title('DHCN Simulation: Multi-Agent Synchronization with Phase 4 Memory Growth')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('dhcn_simulation_final.png', dpi=300)
print("Plot saved as dhcn_simulation_final.png")

# -------------------------------
# CSV Export
# -------------------------------
with open('dhcn_simulation_final.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([f'Agent_{i+1}' for i in range(num_agents)] + ['Phase', 'CollapseEvent', 'AI_Memory'])
    for t in range(num_steps):
        collapse_flag = 1 if t in collapse_events else 0
        writer.writerow(list(history[t]) + [int(phase_history[t]), collapse_flag, ai_memory_history[t]])
print("Simulation data exported to dhcn_simulation_final.csv")

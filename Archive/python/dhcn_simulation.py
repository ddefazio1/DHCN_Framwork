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
collapse_threshold = 0.02  # Variance threshold to trigger collective decision

# -------------------------------
# Initialize agents
# -------------------------------
agents = np.random.rand(num_agents)
history = np.zeros((num_steps, num_agents))
phase_history = np.zeros(num_steps)
collapse_events = []

# -------------------------------
# Simulation loop
# -------------------------------
for t in range(num_steps):
    # Determine phase
    phase = min(t // phase_duration + 1, 4)
    phase_history[t] = phase
    
    # AI calculates shared insight
    shared_insight = np.mean(agents)
    
    # Update each agent
    for i in range(num_agents):
        influence = ai_base_influence * (2 ** (phase - 1))
        agents[i] += influence * (shared_insight - agents[i]) + noise_level * np.random.randn()
        agents[i] = np.clip(agents[i], 0, 1)
    
    history[t] = agents
    
    # Check for collective decision (variance below threshold)
    variance = np.var(agents)
    if variance < collapse_threshold and (len(collapse_events) == 0 or t > collapse_events[-1] + 1):
        collapse_events.append(t)

# -------------------------------
# Visualization: Agent States + Collapses
# -------------------------------
plt.figure(figsize=(12,6))
phase_colors = {1:'#a6cee3', 2:'#1f78b4', 3:'#b2df8a', 4:'#33a02c'}

for i in range(num_agents):
    plt.plot(history[:,i], label=f'Agent {i+1}', lw=1.2)

# Highlight phases
for phase in range(1,5):
    start = (phase-1)*phase_duration
    end = min(phase*phase_duration, num_steps)
    plt.axvspan(start, end, color=phase_colors[phase], alpha=0.1, label=f'Phase {phase}' if i==0 else "")

# Mark collapse events
for t in collapse_events:
    plt.axvline(t, color='red', linestyle='--', lw=1.5, label='Collective Decision' if t==collapse_events[0] else "")

plt.xlabel('Time Step')
plt.ylabel('Agent State')
plt.title('DHCN Simulation: Multi-Agent AI-Mediated Synchronization with Decision Collapse')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# -------------------------------
# Export to CSV
# -------------------------------
export_csv = True
if export_csv:
    with open('dhcn_simulation_collapse.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f'Agent_{i+1}' for i in range(num_agents)] + ['Phase', 'CollapseEvent'])
        for t in range(num_steps):
            collapse_flag = 1 if t in collapse_events else 0
            writer.writerow(list(history[t]) + [int(phase_history[t]), collapse_flag])
    print("Simulation data exported to dhcn_simulation_collapse.csv")

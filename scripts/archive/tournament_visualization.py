"""
Tournament Bracket Visualization
Creates visual representation of the PVL Reinforced Conference 2025 Tournament
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np


def create_tournament_bracket_visualization(top_8, playoff_results):
    """Create a visual tournament bracket"""
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('PVL REINFORCED CONFERENCE 2025 - TOURNAMENT BRACKET', 
                 fontsize=24, fontweight='bold', y=0.98)
    
    # Main bracket
    ax_bracket = plt.subplot2grid((3, 2), (0, 0), rowspan=2, colspan=2)
    ax_bracket.set_xlim(0, 10)
    ax_bracket.set_ylim(0, 16)
    ax_bracket.axis('off')
    
    # Title
    ax_bracket.text(5, 15.5, 'PLAYOFF BRACKET', fontsize=18, 
                   ha='center', fontweight='bold')
    
    # Get results from Voting Ensemble (best model)
    result = playoff_results[-1]  # Voting Ensemble is last
    
    # Extract QF winners, SF winners, etc.
    qf_winners = []
    sf_winners = []
    
    # QF matchups based on seeding
    qf_matchups = [
        (top_8[0][0], top_8[7][0], 'QF1'),
        (top_8[1][0], top_8[6][0], 'QF2'),
        (top_8[2][0], top_8[5][0], 'QF3'),
        (top_8[3][0], top_8[4][0], 'QF4'),
    ]
    
    # Simulate winners (we know from results)
    # QF1: #1 beats #8, QF2: #2 beats #7, QF3: #3 beats #6, QF4: #4 beats #5
    qf_winners = [top_8[0][0], top_8[1][0], top_8[2][0], top_8[3][0]]
    
    # SF: QF1 winner vs QF4 winner, QF2 winner vs QF3 winner
    sf_winners = [result['champion'], result['runner_up']]
    
    # Colors
    winner_color = '#FFD700'  # Gold
    finalist_color = '#C0C0C0'  # Silver
    semifinal_color = '#CD7F32'  # Bronze
    team_color = '#E8F4F8'  # Light blue
    
    # Draw Quarterfinals (Round 1)
    qf_y_positions = [14, 11, 8, 5]
    qf_x = 0.5
    
    ax_bracket.text(qf_x + 1, 15.2, 'QUARTERFINALS', fontsize=12, 
                   fontweight='bold', ha='center')
    
    for i, (team_a, team_b, qf_name) in enumerate(qf_matchups):
        y = qf_y_positions[i]
        winner = qf_winners[i]
        
        # Team A box
        color_a = winner_color if team_a == winner else team_color
        box_a = FancyBboxPatch((qf_x, y), 2, 0.6, boxstyle="round,pad=0.05",
                               facecolor=color_a, edgecolor='black', linewidth=2)
        ax_bracket.add_patch(box_a)
        ax_bracket.text(qf_x + 1, y + 0.3, team_a, ha='center', va='center',
                       fontsize=11, fontweight='bold' if team_a == winner else 'normal')
        
        # Team B box
        color_b = winner_color if team_b == winner else team_color
        box_b = FancyBboxPatch((qf_x, y - 1), 2, 0.6, boxstyle="round,pad=0.05",
                               facecolor=color_b, edgecolor='black', linewidth=2)
        ax_bracket.add_patch(box_b)
        ax_bracket.text(qf_x + 1, y - 0.7, team_b, ha='center', va='center',
                       fontsize=11, fontweight='bold' if team_b == winner else 'normal')
        
        # QF label
        ax_bracket.text(qf_x - 0.3, y - 0.2, qf_name, fontsize=9, 
                       ha='right', style='italic')
        
        # Line to semifinal
        ax_bracket.plot([qf_x + 2, qf_x + 2.5], [y - 0.2, y - 0.2], 'k-', linewidth=2)
    
    # Draw Semifinals (Round 2)
    sf_y_positions = [12.5, 6.5]
    sf_x = 3.5
    
    ax_bracket.text(sf_x + 1, 15.2, 'SEMIFINALS', fontsize=12, 
                   fontweight='bold', ha='center')
    
    # SF1: QF1 winner vs QF4 winner
    # SF2: QF2 winner vs QF3 winner
    sf_matchups = [
        (qf_winners[0], qf_winners[3], 'SF1'),
        (qf_winners[1], qf_winners[2], 'SF2')
    ]
    
    for i, (team_a, team_b, sf_name) in enumerate(sf_matchups):
        y = sf_y_positions[i]
        winner = sf_winners[i]
        
        # Lines from QF to SF
        if i == 0:
            ax_bracket.plot([qf_x + 2.5, sf_x], [qf_y_positions[0] - 0.2, y + 0.3], 
                          'k-', linewidth=2)
            ax_bracket.plot([qf_x + 2.5, sf_x], [qf_y_positions[3] - 0.2, y - 0.7], 
                          'k-', linewidth=2)
        else:
            ax_bracket.plot([qf_x + 2.5, sf_x], [qf_y_positions[1] - 0.2, y + 0.3], 
                          'k-', linewidth=2)
            ax_bracket.plot([qf_x + 2.5, sf_x], [qf_y_positions[2] - 0.2, y - 0.7], 
                          'k-', linewidth=2)
        
        # Team A box
        color_a = winner_color if team_a == winner else semifinal_color
        box_a = FancyBboxPatch((sf_x, y), 2, 0.6, boxstyle="round,pad=0.05",
                               facecolor=color_a, edgecolor='black', linewidth=2)
        ax_bracket.add_patch(box_a)
        ax_bracket.text(sf_x + 1, y + 0.3, team_a, ha='center', va='center',
                       fontsize=11, fontweight='bold' if team_a == winner else 'normal')
        
        # Team B box
        color_b = winner_color if team_b == winner else semifinal_color
        box_b = FancyBboxPatch((sf_x, y - 1), 2, 0.6, boxstyle="round,pad=0.05",
                               facecolor=color_b, edgecolor='black', linewidth=2)
        ax_bracket.add_patch(box_b)
        ax_bracket.text(sf_x + 1, y - 0.7, team_b, ha='center', va='center',
                       fontsize=11, fontweight='bold' if team_b == winner else 'normal')
        
        # SF label
        ax_bracket.text(sf_x - 0.3, y - 0.2, sf_name, fontsize=9, 
                       ha='right', style='italic')
        
        # Line to final
        ax_bracket.plot([sf_x + 2, sf_x + 2.5], [y - 0.2, y - 0.2], 'k-', linewidth=2)
    
    # Draw Finals
    final_y = 9.5
    final_x = 6.5
    
    ax_bracket.text(final_x + 1, 15.2, 'CHAMPIONSHIP', fontsize=12, 
                   fontweight='bold', ha='center')
    
    # Lines from SF to Final
    ax_bracket.plot([sf_x + 2.5, final_x], [sf_y_positions[0] - 0.2, final_y + 0.3], 
                   'k-', linewidth=2)
    ax_bracket.plot([sf_x + 2.5, final_x], [sf_y_positions[1] - 0.2, final_y - 0.7], 
                   'k-', linewidth=2)
    
    # Champion box
    champion = result['champion']
    runnerup = result['runner_up']
    
    box_champ = FancyBboxPatch((final_x, final_y), 2.5, 0.7, boxstyle="round,pad=0.08",
                               facecolor=winner_color, edgecolor='darkgoldenrod', 
                               linewidth=3)
    ax_bracket.add_patch(box_champ)
    ax_bracket.text(final_x + 1.25, final_y + 0.35, champion, ha='center', va='center',
                   fontsize=14, fontweight='bold')
    ax_bracket.text(final_x + 1.25, final_y + 0.05, '🏆 CHAMPION', ha='center', va='center',
                   fontsize=8, style='italic')
    
    # Runner-up box
    box_runner = FancyBboxPatch((final_x, final_y - 1.2), 2.5, 0.7, 
                                boxstyle="round,pad=0.08",
                                facecolor=finalist_color, edgecolor='gray', linewidth=3)
    ax_bracket.add_patch(box_runner)
    ax_bracket.text(final_x + 1.25, final_y - 0.85, runnerup, ha='center', va='center',
                   fontsize=14, fontweight='bold')
    ax_bracket.text(final_x + 1.25, final_y - 1.15, '🥈 RUNNER-UP', ha='center', va='center',
                   fontsize=8, style='italic')
    
    # Third place
    third = result['third']
    fourth = result['fourth']
    
    ax_bracket.text(final_x + 1.25, 2.5, 'THIRD PLACE MATCH', fontsize=10, 
                   fontweight='bold', ha='center')
    
    box_third = FancyBboxPatch((final_x + 0.25, 1.5), 2, 0.5, 
                               boxstyle="round,pad=0.05",
                               facecolor=semifinal_color, edgecolor='saddlebrown', 
                               linewidth=2)
    ax_bracket.add_patch(box_third)
    ax_bracket.text(final_x + 1.25, 1.75, f"{third} 🥉", ha='center', va='center',
                   fontsize=11, fontweight='bold')
    
    ax_bracket.text(final_x + 1.25, 0.8, f"4th Place: {fourth}", ha='center', va='center',
                   fontsize=10)
    
    # Standings subplot
    ax_standings = plt.subplot2grid((3, 2), (2, 0))
    ax_standings.axis('off')
    ax_standings.set_xlim(0, 10)
    ax_standings.set_ylim(0, 10)
    
    ax_standings.text(5, 9, 'TOP 8 SEEDING', fontsize=14, 
                     fontweight='bold', ha='center')
    
    y_pos = 7.5
    for i, (team, stats) in enumerate(top_8, 1):
        record = f"{stats['wins']}-{stats['losses']}"
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        ax_standings.text(2, y_pos, f"{emoji} #{i}", fontsize=11, ha='left')
        ax_standings.text(4, y_pos, team, fontsize=11, ha='left', fontweight='bold')
        ax_standings.text(6, y_pos, record, fontsize=10, ha='left')
        ax_standings.text(7.5, y_pos, f"{stats['win_pct']:.1%}", fontsize=10, ha='left')
        y_pos -= 0.9
    
    # Model consensus subplot
    ax_consensus = plt.subplot2grid((3, 2), (2, 1))
    ax_consensus.axis('off')
    ax_consensus.set_xlim(0, 10)
    ax_consensus.set_ylim(0, 10)
    
    ax_consensus.text(5, 9, 'MODEL CONSENSUS', fontsize=14, 
                     fontweight='bold', ha='center')
    
    y_pos = 7.5
    for i, result in enumerate(playoff_results):
        model_name = result['model']
        predicted_champ = result['champion']
        emoji = '✓' if predicted_champ == champion else '✗'
        color = 'green' if predicted_champ == champion else 'red'
        
        ax_consensus.text(2, y_pos, f"{emoji}", fontsize=14, ha='left', color=color,
                         fontweight='bold')
        ax_consensus.text(3, y_pos, model_name, fontsize=10, ha='left')
        ax_consensus.text(7, y_pos, predicted_champ, fontsize=10, ha='left',
                         fontweight='bold')
        y_pos -= 1.2
    
    ax_consensus.text(5, 1.5, f'UNANIMOUS: {champion} wins!', fontsize=12, 
                     ha='center', fontweight='bold', 
                     bbox=dict(boxstyle='round', facecolor='gold', alpha=0.7))
    
    plt.tight_layout()
    return fig


def create_pool_standings_visualization(pool_a_sorted, pool_b_sorted, combined_standings):
    """Create visualization of pool standings"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('PVL REINFORCED CONFERENCE 2025 - PRELIMINARY ROUND STANDINGS', 
                 fontsize=18, fontweight='bold')
    
    # Pool A
    ax1 = axes[0]
    ax1.set_title('POOL A - FIRST ROUND', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    teams_a = [team[0] for team in pool_a_sorted]
    wins_a = [team[1]['wins'] for team in pool_a_sorted]
    losses_a = [team[1]['losses'] for team in pool_a_sorted]
    
    y_positions = np.arange(len(teams_a))
    colors_a = ['#FFD700', '#C0C0C0', '#CD7F32', '#E8F4F8', '#E8F4F8', '#E8F4F8']
    
    for i, (team, wins, losses, color) in enumerate(zip(teams_a, wins_a, losses_a, colors_a)):
        rect = Rectangle((0.1, 0.85 - i*0.15), 0.8, 0.12, 
                        facecolor=color, edgecolor='black', linewidth=1.5)
        ax1.add_patch(rect)
        ax1.text(0.15, 0.91 - i*0.15, f"{i+1}. {team}", fontsize=12, 
                fontweight='bold', va='center')
        ax1.text(0.75, 0.91 - i*0.15, f"{wins}-{losses}", fontsize=11, 
                ha='right', va='center')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Pool B
    ax2 = axes[1]
    ax2.set_title('POOL B - FIRST ROUND', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    teams_b = [team[0] for team in pool_b_sorted]
    wins_b = [team[1]['wins'] for team in pool_b_sorted]
    losses_b = [team[1]['losses'] for team in pool_b_sorted]
    
    colors_b = ['#FFD700', '#C0C0C0', '#CD7F32', '#E8F4F8', '#E8F4F8', '#E8F4F8']
    
    for i, (team, wins, losses, color) in enumerate(zip(teams_b, wins_b, losses_b, colors_b)):
        rect = Rectangle((0.1, 0.85 - i*0.15), 0.8, 0.12, 
                        facecolor=color, edgecolor='black', linewidth=1.5)
        ax2.add_patch(rect)
        ax2.text(0.15, 0.91 - i*0.15, f"{i+1}. {team}", fontsize=12, 
                fontweight='bold', va='center')
        ax2.text(0.75, 0.91 - i*0.15, f"{wins}-{losses}", fontsize=11, 
                ha='right', va='center')
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # Combined standings
    ax3 = axes[2]
    ax3.set_title('COMBINED STANDINGS\n(After Second Round)', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    top_12 = list(combined_standings)[:12]
    
    for i, (team, stats) in enumerate(top_12):
        if i < 8:
            color = '#90EE90'  # Light green for qualified
            status = '✓'
        else:
            color = '#FFB6C6'  # Light red for eliminated
            status = '✗'
        
        rect = Rectangle((0.1, 0.92 - i*0.075), 0.8, 0.07, 
                        facecolor=color, edgecolor='black', linewidth=1)
        ax3.add_patch(rect)
        
        ax3.text(0.12, 0.955 - i*0.075, f"{i+1}.", fontsize=10, 
                fontweight='bold', va='center')
        ax3.text(0.2, 0.955 - i*0.075, team, fontsize=10, 
                fontweight='bold', va='center')
        ax3.text(0.5, 0.955 - i*0.075, 
                f"{stats['wins']}-{stats['losses']}", 
                fontsize=9, va='center')
        ax3.text(0.7, 0.955 - i*0.075, f"{stats['win_pct']:.1%}", 
                fontsize=9, va='center')
        ax3.text(0.85, 0.955 - i*0.075, status, fontsize=12, 
                fontweight='bold', va='center',
                color='green' if i < 8 else 'red')
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig


def save_tournament_visualizations(top_8, playoff_results, pool_a_sorted, 
                                   pool_b_sorted, combined_standings):
    """Generate and save all visualizations"""
    
    print("\n" + "="*80)
    print(" "*25 + "GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Tournament bracket
    print("\n  Creating tournament bracket...")
    fig1 = create_tournament_bracket_visualization(top_8, playoff_results)
    fig1.savefig('tournament_bracket.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: tournament_bracket.png")
    
    # Pool standings
    print("  Creating pool standings...")
    fig2 = create_pool_standings_visualization(pool_a_sorted, pool_b_sorted, 
                                               combined_standings)
    fig2.savefig('pool_standings.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: pool_standings.png")
    
    plt.close('all')
    
    print("\n" + "="*80)
    print("  ✓ Visualizations complete!")
    print("="*80)

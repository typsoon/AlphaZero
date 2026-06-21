import json
import os
import sys

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Evaluation Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
%STYLE_CSS%
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Performance Report</h1>
            <p>Connect4 AlphaZero Network Evaluation</p>
        </header>

        <div class="summary-cards" id="summaryContainer">
            <!-- Populated via JS -->
        </div>

        <div class="controls">
            <input type="text" id="searchInput" placeholder="Search by test file name...">
            <select id="categoryFilter">
                <option value="all">All Categories</option>
            </select>
            <select id="statusFilter">
                <option value="all">All Statuses</option>
                <option value="pass">Passed</option>
                <option value="fail">Failed</option>
            </select>
        </div>

        <div class="test-list" id="testList">
            <!-- Populated via JS -->
        </div>
    </div>

    <script>
        const resultsData = %RESULTS_JSON%;

        function init() {
            renderSummary();
            populateCategories();
            renderTestList(resultsData);
            setupEventListeners();
        }

        function renderSummary() {
            const total = resultsData.length;
            const passed = resultsData.filter(r => r.passed).length;
            const failed = total - passed;
            const passRate = total === 0 ? 0 : Math.round((passed / total) * 100);

            const container = document.getElementById('summaryContainer');
            container.innerHTML = `
                <div class="card animate-fade-in" style="animation-delay: 0.1s">
                    <h3>Total Tests</h3>
                    <p class="text-primary">${total}</p>
                </div>
                <div class="card animate-fade-in" style="animation-delay: 0.2s">
                    <h3>Passed</h3>
                    <p class="text-success">${passed}</p>
                </div>
                <div class="card animate-fade-in" style="animation-delay: 0.3s">
                    <h3>Failed</h3>
                    <p class="text-danger">${failed}</p>
                </div>
                <div class="card animate-fade-in" style="animation-delay: 0.4s">
                    <h3>Pass Rate</h3>
                    <p class="${passRate >= 80 ? 'text-success' : passRate >= 50 ? 'text-primary' : 'text-danger'}">${passRate}%</p>
                </div>
            `;
        }

        function populateCategories() {
            const categories = [...new Set(resultsData.map(r => r.category))];
            const select = document.getElementById('categoryFilter');
            categories.forEach(cat => {
                const option = document.createElement('option');
                option.value = cat;
                option.textContent = cat;
                select.appendChild(option);
            });
        }

        function renderBoard(board) {
            let html = '<div class="connect4-board">';
            for (let r = 0; r < 6; r++) {
                for (let c = 0; c < 7; c++) {
                    const cell = board[r] ? (board[r][c] || 0) : 0;
                    let cellClass = 'cell';
                    if (cell === 1) cellClass += ' player1';
                    else if (cell === 2) cellClass += ' player2';
                    html += `<div class="${cellClass}"></div>`;
                }
            }
            html += '</div>';
            return html;
        }

        function renderPolicy(policy) {
            let html = '';
            policy.forEach((prob, idx) => {
                const percentage = (prob * 100).toFixed(1);
                html += `
                    <div class="policy-bar-container">
                        <div class="policy-label">${idx}</div>
                        <div class="policy-track">
                            <div class="policy-fill" data-target="${percentage}%"></div>
                        </div>
                        <div class="policy-value">${percentage}%</div>
                    </div>
                `;
            });
            return html;
        }

        function renderTestList(data) {
            const container = document.getElementById('testList');
            container.innerHTML = '';

            if (data.length === 0) {
                container.innerHTML = '<div class="card" style="text-align:center; padding:3rem;"><h3>No tests found matching criteria.</h3></div>';
                return;
            }

            data.forEach((test, index) => {
                const item = document.createElement('div');
                item.className = 'test-item animate-fade-in';
                item.style.animationDelay = `${Math.min(index * 0.05, 0.5)}s`;
                
                const expectedStr = Array.isArray(test.expected_moves) ? test.expected_moves.join(', ') : test.expected_moves;
                const policyHtml = renderPolicy(test.network_policy);
                const boardHtml = renderBoard(test.board);

                let count1 = 0;
                let count2 = 0;
                for (let r = 0; r < 6; r++) {
                    for (let c = 0; c < 7; c++) {
                        const cell = test.board[r] ? (test.board[r][c] || 0) : 0;
                        if (cell === 1) count1++;
                        else if (cell === 2) count2++;
                    }
                }
                const playerToMove = count1 === count2 ? "Player 1 (Red)" : "Player 2 (Yellow)";

                item.innerHTML = `
                    <div class="test-header">
                        <div class="test-info">
                            <span class="status-badge ${test.passed ? 'pass' : 'fail'}">
                                ${test.passed ? '✓ PASS' : '✗ FAIL'}
                            </span>
                            <span class="test-title">${test.test_name}</span>
                            <span class="test-category">${test.category}</span>
                        </div>
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="transition: transform 0.3s; color: #94a3b8;"><polyline points="6 9 12 15 18 9"></polyline></svg>
                    </div>
                    <div class="test-details">
                        <div class="details-grid">
                            <div class="details-section">
                                <h4>Evaluation Details</h4>
                                <div class="data-row">
                                    <span style="color: #94a3b8">Expected Move(s):</span>
                                    <span style="font-weight: 600">${expectedStr}</span>
                                </div>
                                <div class="data-row">
                                    <span style="color: #94a3b8">Chosen Move:</span>
                                    <span style="font-weight: 600; color: ${test.passed ? 'var(--success)' : 'var(--danger)'}">${test.chosen_move}</span>
                                </div>
                                <div class="data-row">
                                    <span style="color: #94a3b8">Network Value:</span>
                                    <span style="font-weight: 600">${test.network_value !== undefined ? test.network_value.toFixed(4) : "N/A"}</span>
                                </div>
                                <div class="data-row" style="margin-top: 1rem;">
                                    <span style="color: #94a3b8">Player to Move:</span>
                                    <span style="font-weight: 600">${playerToMove}</span>
                                </div>
                                
                                <h4 style="margin-top: 2rem;">Network Policy</h4>
                                <div class="policy-wrapper">
                                    ${policyHtml}
                                </div>
                            </div>
                            <div class="details-section">
                                <h4>Board State</h4>
                                <div class="board-container">
                                    ${boardHtml}
                                </div>
                            </div>
                        </div>
                    </div>
                `;

                // Add toggle behavior
                const header = item.querySelector('.test-header');
                header.addEventListener('click', () => {
                    const isActive = item.classList.contains('active');
                    
                    if (!isActive) {
                        item.classList.add('active');
                        item.querySelector('svg').style.transform = 'rotate(180deg)';
                        
                        // Animate policy bars
                        setTimeout(() => {
                            item.querySelectorAll('.policy-fill').forEach(fill => {
                                fill.style.width = fill.getAttribute('data-target');
                            });
                        }, 50);
                    } else {
                        item.classList.remove('active');
                        item.querySelector('svg').style.transform = 'rotate(0deg)';
                        
                        // Reset policy bars
                        item.querySelectorAll('.policy-fill').forEach(fill => {
                            fill.style.width = '0%';
                        });
                    }
                });

                container.appendChild(item);
            });
        }

        function setupEventListeners() {
            const searchInput = document.getElementById('searchInput');
            const categoryFilter = document.getElementById('categoryFilter');
            const statusFilter = document.getElementById('statusFilter');

            function filterData() {
                const search = searchInput.value.toLowerCase();
                const category = categoryFilter.value;
                const status = statusFilter.value;

                const filtered = resultsData.filter(test => {
                    const matchSearch = test.test_name.toLowerCase().includes(search);
                    const matchCategory = category === 'all' || test.category === category;
                    let matchStatus = true;
                    if (status === 'pass') matchStatus = test.passed === true;
                    if (status === 'fail') matchStatus = test.passed === false;

                    return matchSearch && matchCategory && matchStatus;
                });

                renderTestList(filtered);
            }

            searchInput.addEventListener('input', filterData);
            categoryFilter.addEventListener('change', filterData);
            statusFilter.addEventListener('change', filterData);
        }

        // Initialize on load
        document.addEventListener('DOMContentLoaded', init);
    </script>
</body>
</html>
"""


def generate_report(json_path, output_path):
    if not os.path.exists(json_path):
        print(f"Error: Could not find '{json_path}'.")
        sys.exit(1)

    with open(json_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: '{json_path}' is not a valid JSON file.")
            sys.exit(1)

    json_str = json.dumps(data)

    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    with open(css_path, "r") as f:
        css_content = f.read()

    html_content = HTML_TEMPLATE.replace("%RESULTS_JSON%", json_str).replace(
        "%STYLE_CSS%", css_content
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"Successfully generated report at '{output_path}'.")


if __name__ == "__main__":
    default_json = os.path.join(os.path.dirname(__file__), "results.json")
    default_html = os.path.join(os.path.dirname(__file__), "performance_report.html")

    # Allow overriding default paths via CLI arguments
    json_path = sys.argv[1] if len(sys.argv) > 1 else default_json
    html_path = sys.argv[2] if len(sys.argv) > 2 else default_html

    generate_report(json_path, html_path)

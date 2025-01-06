#include <iostream>
#include <vector>
#include <unordered_map>
#include <random>
#include <tuple>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <string>
#include <thread>
#include <utility>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <fstream>

// Constants for the environment and learning
const int grid_size = 100;
int num_agents = grid_size /5;
const int num_episodes = 1000;
const int max_steps = 10000;
const double learning_rate = 0.5;
const double discount_factor = 0.9;
double epsilon = 1.0;
const double max_epsilon = 1.0;
const double min_epsilon = 0.01;
const double epsilon_decay = 0.995;

// Pointers to the constants
const int* grid_size_ptr = &grid_size;
const int* num_episodes_ptr = &num_episodes;
const int* max_steps_ptr = &max_steps;
const double* learning_rate_ptr = &learning_rate;
const double* discount_factor_ptr = &discount_factor;
double* epsilon_ptr = &epsilon;
const double* max_epsilon_ptr = &max_epsilon;
const double* min_epsilon_ptr = &min_epsilon;
const double* epsilon_decay_ptr = &epsilon_decay;

// Directions for actions (up, down, left, right)
std::vector<std::pair<int, int>> actions = {
    {-1, 0}, // Move up
    {1, 0},  // Move down
    {0, -1}, // Move left
    {0, 1}   // Move right
};

// Offsets for the 8 surrounding squares 
std::vector<std::pair<int, int>> offsets = {
    {-1, -1}, {-1, 0}, {-1, 1}, // Top-left, Top, Top-right
    { 0, -1},          { 0, 1}, // Left,    Right
    { 1, -1}, { 1, 0}, { 1, 1} // Bottom-left, Bottom, Bottom-right
};

// Random number generator setup
std::default_random_engine generator(std::time(nullptr));
std::uniform_real_distribution<double> distribution(0.0, 1.0);
std::uniform_int_distribution<int> random_position(0, *grid_size_ptr - 1);

// Agent class
class Agent 
{
public:
    std::pair<int, int> position;
    double total_reward = 0;
    double highest_point = 0;
    double lowest_point = 0;
    std::unordered_map<std::string, std::vector<double>> q_table;

    std::pair<int, int>* position_ptr = &position;
    double* total_reward_ptr = &total_reward;
    double* highest_point_ptr = &highest_point;
    double* lowest_point_ptr = &lowest_point;
    std::unordered_map<std::string, std::vector<double>>* q_table_ptr = &q_table;


    Agent(std::pair<int, int> start_pos):
        position(start_pos) {}

    std::string stateToKey(const std::vector<int>& state)
    {
        std::string key;
        for (int val : state)
        {
            key += std::to_string(val) + ",";
        }
        return key;
    }

    int chooseAction(const std::string& state_key)
    {
        if (distribution(generator) < *epsilon_ptr) 
        {
            return std::uniform_int_distribution<int>(0, actions.size() - 1)(generator); // Explore
        }
        else 
        {
            if (q_table.find(state_key) == q_table.end()) 
            {
                q_table[state_key] = std::vector<double>(actions.size(), 0.0); // Initialize unseen state
            }
            return std::max_element(q_table[state_key].begin(), q_table[state_key].end()) - q_table[state_key].begin(); // Exploit
        }
    }

    void updateQValue(const std::string& state_key, int action, int reward, const std::string& new_state_key) 
    {
        if (q_table.find(state_key) == q_table.end())
        {
            q_table[state_key] = std::vector<double>(actions.size(), 0.0);
        }
        if (q_table.find(new_state_key) == q_table.end()) 
        {
            q_table[new_state_key] = std::vector<double>(actions.size(), 0.0);
        }
        q_table[state_key][action] += *learning_rate_ptr * (reward + *discount_factor_ptr * *std::max_element(q_table[new_state_key].begin(), q_table[new_state_key].end()) - q_table[state_key][action]);
    }
};

// Function to visualize the grid with all agents
void visualize_grid(const std::vector<std::vector<int>>& grid, const std::vector<Agent>& agents) {
    std::vector<std::vector<std::string>> display_grid(*grid_size_ptr, std::vector<std::string>(*grid_size_ptr, "."));

    for (int i = 0; i < *grid_size_ptr; ++i) 
    {
        for (int j = 0; j < *grid_size_ptr; ++j)
        {
            if (grid[i][j] == 100) 
            {
                display_grid[i][j] = "R"; // Reward
            }
            else if (grid[i][j] == -100) 
            {
                display_grid[i][j] = "H"; // Hazard
            }
            else if (grid[i][j] == -999)
            {
                display_grid[i][j] = "X"; // Obstacle
            }
        }
    }

    for (size_t i = 0; i < agents.size(); ++i) 
    {
        int x = agents[i].position.first;
        int y = agents[i].position.second;
        display_grid[x][y] = "A" + std::to_string(i + 1);
    }

#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif

    for (int i = 0; i < *grid_size_ptr; ++i) 
    {
        for (int j = 0; j < *grid_size_ptr; ++j) 
        {
            std::cout << display_grid[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Function to create unique keys for states
std::string stateToKey(const std::vector<int>& state) 
{
    std::string key;
    for (int val : state) 
    {
        key += std::to_string(val) + ",";
    }
    return key;
}

void initialize_chunk(std::vector<std::vector<int>>& grid, int start_row, int end_row)
{
    for (int i = start_row; i < end_row; ++i)
    {
        for (int j = 0; j < grid[i].size(); ++j)
        {
            double rand_value = distribution(generator);
            if (rand_value < 0.33)
            {
                grid[i][j] = (distribution(generator) < 0.5) ? 100 : -100;
            }
            else if (rand_value < 0.43)
            {
                grid[i][j] = -999;
            } 
            else
            {
                grid[i][j] = 0;
            }
        }
    }
}
// Function to initialize the grid with random rewards and hazards
std::vector<std::vector<int>> initialize_grid() {
    std::vector<std::vector<int>> grid(*grid_size_ptr, std::vector<int>(*grid_size_ptr, 0));

    int num_threads = std::thread::hardware_concurrency();
    int chunk_size = grid_size / num_threads;
    int remainder = grid_size % num_threads;

    std::vector<std::thread> threads;
    int start_row = 0;

    for (int t = 0; t < num_threads; ++t)
    {
        int end_row = start_row + chunk_size + (t < remainder ? 1 : 0);
        threads.emplace_back(initialize_chunk, std::ref(grid), start_row, end_row);
        start_row = end_row;
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    return grid;
}

// Function to get surrounding tiles as a vector
std::vector<int> get_surrounding_tiles(const std::vector<std::vector<int>>& grid, std::pair<int, int> position)
{
    int grid_size = grid.size();
    int x = position.first;
    int y = position.second;

    // Precomputed offsets for a 3x3 neighborhood excluding the center
    const std::vector<std::pair<int, int>> offsets = {
        {-1, -1}, {-1, 0}, {-1, 1},{ 0, -1}, { 0,  1},{ 1, -1}, { 1,  0}, { 1,  1}
    };

    std::vector<int> surrounding;
    surrounding.reserve(24); // Reserve space for 24 elements

    for (const auto& offset : offsets)
    {
        int nx = (x + offset.first + grid_size) % grid_size;
        int ny = (y + offset.second + grid_size) % grid_size;
        surrounding.push_back(grid[nx][ny]);
    }

    return surrounding;
}

struct PairHash
{
    size_t operator()(const std::pair<int, int>& p) const
    {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

// Function to handle out-of-bounds movement (infinite grid)
std::pair<int, int> handle_out_of_bounds(std::pair<int, int> state)
{
    return { (state.first + *grid_size_ptr) % *grid_size_ptr, (state.second + *grid_size_ptr) % *grid_size_ptr };
}

// Function to find the index of an agent by coordinates
int find_agent_index_by_coordinates(const std::vector<Agent>& agents, int target_x, int target_y) {
    auto it = std::find_if(agents.begin(), agents.end(), [target_x, target_y](const Agent& agent) {
        return agent.position.first == target_x && agent.position.second == target_y;
        });

    // If the agent is found, calculate the index
    if (it != agents.end()) {
        return std::distance(agents.begin(), it);
    }
    else {
        return -1; // Return -1 if the agent is not found
    }
}

void update_birth_count_map(const std::vector<std::vector<int>>& grid, const std::vector<Agent>& agents, std::unordered_map<std::pair<int, int>, int, PairHash>& birth_count_map)
{
    for (const auto& agent : agents)
    {
        for (const auto& offset : offsets)
        {
            std::pair<int, int> neighbor = handle_out_of_bounds({ agent.position.first + offset.first, agent.position.second + offset.second });
            if (grid[neighbor.first][neighbor.second] == 0 && *agent.total_reward_ptr > 0)
            {
                birth_count_map[neighbor]++;
            }
        }
    }
}

void update_death_count_map(const std::vector<Agent>& agents, std::unordered_map<std::pair<int, int>, int, PairHash>& death_count_map)
{
    for (const auto& agent : agents)
    {
        for (const auto& offset : offsets)
        {
            std::pair<int, int> neighbor = handle_out_of_bounds({ agent.position.first + offset.first, agent.position.second + offset.second });
            int index = find_agent_index_by_coordinates(agents, neighbor.first, neighbor.second);
            if (index != -1)
            {
                death_count_map[neighbor]++;
            }
            
        }
    }
}

std::vector<std::pair<int, int>> find_valid_squares(const std::unordered_map<std::pair<int, int>, int, PairHash>& count_map, const std::vector<std::vector<int>>& grid)
{
    std::vector<std::pair<int, int>> valid_squares;

    for (auto it = count_map.begin(); it != count_map.end(); ++it)
    {
        const std::pair<int, int>& square = it->first;
        int count = it->second;
        if (count == 2)
        {
            valid_squares.push_back(square);
        }     
    }

    return valid_squares;
}
std::vector<std::pair<int, int>> find_deadly_squares(const std::unordered_map<std::pair<int, int>, int, PairHash>& count_map, std::vector<std::vector<int>>& grid)
{
    std::vector<std::pair<int, int>> deadly_squares;

    for (auto it = count_map.begin(); it != count_map.end(); ++it)
    {
        const std::pair<int, int>& square = it->first;
        int count = it->second;

        if (count > 2)
        {
            grid[square.first][square.second] = 0;
            deadly_squares.push_back(square);
        }
    }

    return deadly_squares;
}

void modify_agents(std::vector<Agent>& agents, const std::vector<std::pair<int, int>>& valid_squares, const std::vector<std::pair<int, int>>& deadly_squares, std::vector<std::vector<int>>& grid)
{
    for (const auto& square : valid_squares)
    {
        Agent new_agent = Agent(square);

        // Precomputed offsets for a 3x3 neighborhood excluding the center
        const std::vector<std::pair<int, int>> offsets = {
            {-1, -1}, {-1, 0}, {-1, 1},{ 0, -1}, { 0,  1},{ 1, -1}, { 1,  0}, { 1,  1}
        };

        int index = 0;

        for (const auto& offset : offsets)
        {
            int nx = (square.first + offset.first + grid_size) % grid_size;
            int ny = (square.second + offset.second + grid_size) % grid_size;
            if (grid[nx][ny] == -50)
            {
                index = find_agent_index_by_coordinates(agents, nx, ny);
            }
        }

        for (const auto& entry : agents[index].q_table)
        {
            const auto& key = entry.first;
            const auto& values = entry.second;
            if (new_agent.q_table.find(key) != new_agent.q_table.end())
            {
                new_agent.q_table[key].insert(new_agent.q_table[key].end(), values.begin(), values.end());
            }
            else
            {
                new_agent.q_table[key] = values;
            }
        }

        grid[square.first][square.second] == -50;
        agents.emplace_back(new_agent);
    }
    for (const auto& square : deadly_squares)
    {
        agents.erase(
            std::remove_if(agents.begin(), agents.end(),
                [&square](const Agent& agent)
                {
                    return agent.position == square;
                }),
            agents.end());
    }

}



bool check_collision(const std::vector<Agent>& agents, const std::pair<int, int>& new_position)
{
    for (const auto& agent : agents)
    {
        if (agent.position == new_position)
        {
            return true;
        }
    }
    return false;
}

void place_agents_on_grid(std::vector<std::vector<int>>& grid, const std::vector<Agent>& agents)
{
    for (const auto& agent : agents)
    {
        grid[agent.position.first][agent.position.second] = -50;
    }
}


int main() 
{
    // Open the CSV file in append mode
    std::ofstream file("test_simulation.csv", std::ios::app);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file for writing.\n";
        return 1;
    }
    file << "Episode,Number of Agents,Total Reward,Average Reward,Highest Point,Lowest Point\n";
    file.close();

    std::vector<std::vector<int>> grid = initialize_grid();
    std::vector<Agent> agents;
    for (int i = 0; i < num_agents; i++)
    {
        agents.push_back(Agent({ random_position(generator), random_position(generator) }));
    }

    for (int episode = 0; episode < num_episodes; ++episode) 
    {
        grid = initialize_grid();

        double total_global_reward = 0;
        double global_lowest_point = 0;
        double global_highest_point = 0;

        for (auto& agent : agents)
        {
            agent.position = { random_position(generator), random_position(generator) };
            *agent.total_reward_ptr = 0;
            *agent.highest_point_ptr = 0;
            *agent.lowest_point_ptr = 0;
        }

        for (int step = 0; step < *max_steps_ptr; ++step) 
        {
            place_agents_on_grid(grid, agents);

            for (auto& agent : agents) 
            {
                std::vector<int> state = get_surrounding_tiles(grid, agent.position);
                std::string state_key = agent.stateToKey(state);

                int action = agent.chooseAction(state_key);
                std::pair<int, int> move = actions[action];
                std::pair<int, int> new_position = handle_out_of_bounds({ agent.position.first + move.first, agent.position.second + move.second });

                if (check_collision(agents, new_position))
                {
                    int reward = grid[agent.position.first][agent.position.second] - 50;
                    std::vector<int> new_state = get_surrounding_tiles(grid, agent.position);
                    std::string new_state_key = agent.stateToKey(new_state);

                    agent.updateQValue(state_key, action, reward, new_state_key);
                    *agent.total_reward_ptr += reward;
                    if (*agent.total_reward_ptr > *agent.highest_point_ptr)
                    {
                        *agent.highest_point_ptr = *agent.total_reward_ptr;
                    }
                    if (*agent.total_reward_ptr < *agent.lowest_point_ptr)
                    {
                        *agent.lowest_point_ptr = *agent.total_reward_ptr;
                    }

                    continue;
                }

                if (grid[new_position.first][new_position.second] == -999)
                {
                    int reward = grid[agent.position.first][agent.position.second] - 10;
                    std::vector<int> new_state = get_surrounding_tiles(grid, agent.position);
                    std::string new_state_key = agent.stateToKey(new_state);

                    agent.updateQValue(state_key, action, reward, new_state_key);
                    *agent.total_reward_ptr += reward;
                    if (*agent.total_reward_ptr > *agent.highest_point_ptr)
                    {
                        *agent.highest_point_ptr = *agent.total_reward_ptr;
                    }
                    if (*agent.total_reward_ptr < *agent.lowest_point_ptr)
                    {
                        *agent.lowest_point_ptr = *agent.total_reward_ptr;
                    }

                    continue;
                }

                int reward = grid[new_position.first][new_position.second];
                std::vector<int> new_state = get_surrounding_tiles(grid, new_position);
                std::string new_state_key = agent.stateToKey(new_state);

                agent.updateQValue(state_key, action, reward, new_state_key);
                *agent.total_reward_ptr += reward;
                if (*agent.total_reward_ptr > *agent.highest_point_ptr)
                {
                    *agent.highest_point_ptr = *agent.total_reward_ptr;
                }
                if (*agent.total_reward_ptr < *agent.lowest_point_ptr)
                {
                    *agent.lowest_point_ptr = *agent.total_reward_ptr;
                }


                 grid[new_position.first][new_position.second] = -50;
               

                // Chance to turn the old position into reward or hazard
                if ((grid[agent.position.first][agent.position.second] == 0 ) && distribution(generator) < 0.33 )
                {
                    grid[agent.position.first][agent.position.second] = (distribution(generator) < 0.5) ? 100 : -100;
                }
                else
                {
                    grid[agent.position.first][agent.position.second] = 0;
                }

                agent.position = new_position;
            }

            // Code responsible for the Game of Life rules, comment them out if you want regular RL

            // Create and update agent count map
            std::unordered_map<std::pair<int, int>, int, PairHash> birth_count_map;
            std::unordered_map<std::pair<int, int>, int, PairHash> death_count_map;
            update_birth_count_map(grid,agents, birth_count_map);
            update_death_count_map(agents, death_count_map);

            std::vector<std::pair<int, int>> deadly_squares = find_deadly_squares(death_count_map, grid);
            std::vector<std::pair<int, int>> valid_sqaures = find_valid_squares(birth_count_map, grid);
            

            // Add new agents to these positions
            modify_agents(agents, valid_sqaures, deadly_squares, grid);
            




            place_agents_on_grid(grid, agents);

            // Option to visualise the grid and pause it after each step for observation
            //visualize_grid(grid, agents);
            //system("pause");
        }

        // Decay epsilon
        *epsilon_ptr = std::max(*min_epsilon_ptr, *epsilon_ptr * *epsilon_decay_ptr);
        for (auto& agent : agents)
        {
            total_global_reward += *agent.total_reward_ptr;
            global_highest_point = std::max(global_highest_point, *agent.highest_point_ptr);
            global_lowest_point = std::min(global_lowest_point, *agent.lowest_point_ptr);
        }
        double average_total_reward = total_global_reward / agents.size();

        std::cout << episode << "\n";
        
        
        // Open the CSV file in append mode
        std::ofstream file("test_simulation.csv", std::ios::app);
        if (!file.is_open())
        {
            std::cerr << "Failed to open file for writing.\n";
            return 1;
        }

        // Write metrics to the file
        file << episode << ","
            << agents.size() << ","
            << total_global_reward << ","
            << average_total_reward << ","
            << global_highest_point << ","
            << global_lowest_point << "\n";

        file.close();
        
    }

    return 0;
}

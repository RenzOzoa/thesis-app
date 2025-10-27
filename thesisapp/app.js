// === CONFIGURATION AND GLOBAL STATE ===
// Fixed depot location: Barangay San Manuel, Puerto Princesa, Philippines
const depotLocation = { lat: 9.7735, lng: 118.7524 }; 
const INITIAL_MAP_ZOOM = 13; // Adjusted zoom for a district view

// --- Heterogeneous Fleet Properties ---
// Capacity is defined as "max delivery points" or "max stops".
// Speed is in km/h.
const VEHICLE_PROPERTIES = {
    motorcycle: { speed: 40, capacity: 10 },
    tricycle: { speed: 35, capacity: 15 }, // Avg speed for 30-40 range
    car: { speed: 40, capacity: 20 },
    van: { speed: 40, capacity: 30 }
};

// Global State
let map;
let markers = [];
let polylines = [];
let currentSimulator = null;
let customerData = [];
let nextCustomerId = 1;

// --- State for selected vehicle and plotting limit ---
let selectedVehicleType = 'motorcycle'; // Default
let maxCustomers = 10; // Default capacity for motorcycle

const colorPalette = [
    '#ef4444', '#f97316', '#eab308', '#22c55e', 
    '#06b6d4', '#3b82f6', '#8b5cf6', '#d946ef',
    '#64748b', '#dc2626', '#34d399', '#f472b6' 
];


// --- CORE DATA STRUCTURES (Models) ---

class Customer {
    /**
     * Represents a customer with delivery requirements and time windows.
     */
    constructor(id, lat, lng, demand, timeWindowStart, timeWindowEnd, serviceTime = 0.5) {
        this.id = id;
        this.lat = lat;
        this.lng = lng;
        this.demand = demand; // Now represents "1 delivery point"
        this.timeWindowStart = timeWindowStart;
        this.timeWindowEnd = timeWindowEnd;
        this.serviceTime = serviceTime;
    }
}

class Vehicle {
    /**
     * Represents a delivery vehicle state during ant construction.
     */
    constructor(id, type, speed, maxStops) {
        this.id = id;
        this.type = type;         // 'motorcycle', 'van', etc.
        this.speed = speed;       // km/h
        this.maxStops = maxStops; // Max number of customers
        this.currentTime = 0;     // Current time in hours from the start of the day
        this.route = [];          // Stores customer IDs (1-indexed)
        this.currentLocationIndex = 0; // 0 for depot
    }

    /** Resets the vehicle's state for a new ant simulation. */
    reset() {
        this.currentTime = 0;
        this.route = [];
        this.currentLocationIndex = 0;
    }
}


// --- UTILITY FUNCTIONS ---

/** Calculates distance between two LatLng objects (in meters) using the Google Maps Geometry Library. */
const distance = (loc1, loc2) => {
    if (typeof google === 'undefined' || !google.maps.geometry) {
        // Fallback distance for robustness
        return 1000; 
    }
    const latLng1 = new google.maps.LatLng(loc1.lat, loc1.lng);
    const latLng2 = new google.maps.LatLng(loc2.lat, loc2.lng);
    // Returns distance in meters
    return google.maps.geometry.spherical.computeDistanceBetween(latLng1, latLng2);
};

/** Generates a random number following a Gaussian (Normal) distribution. */
const randGaussian = (mean = 0, stdDev = 1) => {
    let u = 0, v = 0;
    while (u === 0) u = Math.random(); 
    while (v === 0) v = Math.random();
    let z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return z * stdDev + mean;
};


// --- ACO CVRP CORE LOGIC (Solver) ---

class AcoCvrpSimulator {
    /**
     * Ant Colony Optimization simulator for CVRP with Time Windows and Environmental factors.
     */
    constructor(depotLocation, customers, fleetConfig, params) {
        this.depotLocation = depotLocation;
        this.customers = customers;
        
        // --- Build vehicle fleet ---
        this.vehicles = [];
        let vId = 0;
        for (const [type, count] of Object.entries(fleetConfig)) {
            if (count > 0) {
                const props = VEHICLE_PROPERTIES[type];
                for (let i = 0; i < count; i++) {
                    this.vehicles.push(new Vehicle(vId++, type, props.speed, props.capacity));
                }
            }
        }
        this.numVehicles = this.vehicles.length;
        // --- (End fleet logic) ---

        // ACO Parameters
        this.numAnts = params.numAnts;
        this.numIterations = params.numIterations;
        this.alpha = params.alpha; // Pheromone influence
        this.beta = params.beta;   // Heuristic influence
        this.rho = params.rho;     // Evaporation rate
        this.Q = params.Q;         // Pheromone deposit constant
        this.tau0 = params.tau0 || 0.1; // Initial pheromone value
        
        // Locations (0: Depot, 1..N: Customers)
        this.locations = [depotLocation].concat(this.customers.map(c => ({ lat: c.lat, lng: c.lng })));
        this.numNodes = this.locations.length;
        
        this.distanceMatrix = this._calculateDistanceMatrix();
        
        // Environmental matrices (traffic, weather)
        this.trafficMatrix = this._initializeEnvironmentalMatrix(1.2, 0.8, 0.3);
        this.weatherMatrix = this._initializeEnvironmentalMatrix(1.1, 0.9, 0.5);
        
        // Initialize Pheromone Matrix
        this.pheromoneMatrix = Array(this.numNodes).fill(0).map(() => 
            Array(this.numNodes).fill(this.tau0)
        );
        
        this.bestSolution = null;
        this.bestCost = Infinity;
    }

    /** Calculates the distance matrix between all nodes (Depot and Customers) in meters. */
    _calculateDistanceMatrix() {
        const n = this.numNodes;
        const distMatrix = Array(n).fill(0).map(() => Array(n).fill(0));
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                if (i !== j) {
                    distMatrix[i][j] = distance(this.locations[i], this.locations[j]); 
                }
            }
        }
        return distMatrix;
    }

    /** Creates a random environmental matrix (e.g., traffic, weather impact on travel time). */
    _initializeEnvironmentalMatrix(highFactor, lowFactor, highChance) {
        const env = Array(this.numNodes).fill(0).map(() => Array(this.numNodes).fill(1.0));
        for (let i = 0; i < this.numNodes; i++) {
            for (let j = 0; j < this.numNodes; j++) {
                if (i !== j) {
                    const r = Math.random();
                    if (r < highChance) {
                        env[i][j] = lowFactor + Math.random() * (highFactor - lowFactor);
                    } else {
                        env[i][j] = 1.0;
                    }
                }
            }
        }
        return env;
    }

    /** Calculates travel time in hours, incorporating distance and environmental factors. */
    _calculateTravelTime(i, j, vehicle) {
        const distanceKm = this.distanceMatrix[i][j] / 1000; 
        const baseSpeed = vehicle.speed; // km/h, from vehicle properties
        const baseTimeHours = distanceKm / baseSpeed;
        const actualTimeHours = baseTimeHours * this.trafficMatrix[i][j] * this.weatherMatrix[i][j];
        return actualTimeHours;
    }
    
    /** Calculates the adjusted distance (cost) used for the ACO path calculation. */
    _calculateAdjustedCost(i, j) {
        // Cost = Distance * Traffic Factor * Weather Factor (in adjusted meters)
        return this.distanceMatrix[i][j] * this.trafficMatrix[i][j] * this.weatherMatrix[i][j];
    }

    /** Calculates the heuristic value ($\eta_{ij}$), which is the inverse of the adjusted cost. */
    _calculateHeuristic(i, j) {
        const adjustedCost = this._calculateAdjustedCost(i, j);
        return 1000000.0 / Math.max(adjustedCost, 1); 
    }

    /** Calculates the probabilities for an ant to move from 'current' node to any node in 'unvisited'. */
    _calculateProbability(current, unvisited, vehicle) {
        const probabilities = {};
        let total = 0;

        for (const nextCustomerIndex of unvisited) {
            const customer = this.customers[nextCustomerIndex - 1];
            
            // 1. Capacity Constraint (check number of stops)
            if (vehicle.route.length >= vehicle.maxStops) {
                continue;
            }

            // 2. Time Window Constraint Check
            const travelTime = this._calculateTravelTime(current, nextCustomerIndex, vehicle);
            const arrivalTime = vehicle.currentTime + travelTime;
            let twFactor = 1.0;

            if (arrivalTime > customer.timeWindowEnd) {
                twFactor = 0.001; 
            } else if (arrivalTime < customer.timeWindowStart) {
                twFactor = 0.8; 
            }

            // 3. Probability Calculation (P_ij^k)
            const pheromone = Math.pow(this.pheromoneMatrix[current][nextCustomerIndex], this.alpha);
            const heuristic = Math.pow(this._calculateHeuristic(current, nextCustomerIndex), this.beta);
            
            const probability = pheromone * heuristic * twFactor;
            probabilities[nextCustomerIndex] = probability;
            total += probability;
        }

        // Normalize probabilities
        if (total > 0) {
            for (const customerIndex in probabilities) {
                probabilities[customerIndex] /= total;
            }
        }
        return probabilities;
    }

    /** Selects the next customer index based on the calculated probabilities (Roulette Wheel selection). */
    _selectNextCustomer(probabilities) {
        if (Object.keys(probabilities).length === 0) {
            return -1;
        }
        
        const rand = Math.random();
        let cumulative = 0;
        
        for (const customerIndexStr in probabilities) {
            const customerIndex = parseInt(customerIndexStr);
            cumulative += probabilities[customerIndexStr];
            if (rand <= cumulative) {
                return customerIndex;
            }
        }
        return parseInt(Object.keys(probabilities).pop());
    }

    /** An ant constructs a full set of routes for all available vehicles. */
    _constructAntSolution() {
        let unvisited = new Set(Array.from({ length: this.numNodes - 1 }, (_, i) => i + 1));
        const routes = [];

        // Reset all vehicles for this new ant
        this.vehicles.forEach(v => v.reset());
        
        // Iterate over the fleet (which is now just 1 vehicle)
        for (const vehicle of this.vehicles) {
            if (unvisited.size === 0) break;
            
            let current = 0; // Start at depot (index 0)
            
            while (unvisited.size > 0) {
                const probabilities = this._calculateProbability(current, Array.from(unvisited), vehicle);
                
                if (Object.keys(probabilities).length === 0) {
                    break; 
                }
                
                const nextCustomerIndex = this._selectNextCustomer(probabilities);
                
                if (nextCustomerIndex === -1) {
                    break;
                }
                
                const customer = this.customers[nextCustomerIndex - 1];
                
                // Update vehicle state: add customer
                vehicle.route.push(nextCustomerIndex);
                
                // Time calculation
                const travelTime = this._calculateTravelTime(current, nextCustomerIndex, vehicle);
                const arrivalTime = vehicle.currentTime + travelTime;
                
                // Service start time (wait if early)
                const serviceStart = Math.max(arrivalTime, customer.timeWindowStart);
                vehicle.currentTime = serviceStart + customer.serviceTime;
                
                unvisited.delete(nextCustomerIndex);
                current = nextCustomerIndex;
            }

            // Close route: return to depot (0) if the vehicle served customers
            if (vehicle.route.length > 0) {
                const travelTimeBack = this._calculateTravelTime(current, 0, vehicle);
                vehicle.currentTime += travelTimeBack;
            }
            routes.push(vehicle.route);
        }
        
        return routes;
    }
    
    /** Calculates the total cost (adjusted distance in meters) for a set of routes. */
    _calculateSolutionCost(routes) {
        let totalCost = 0;
        
        for (const route of routes) {
            if (!route || route.length === 0) continue;
            
            let prev = 0; // Depot (index 0)
            for (const customerIndex of route) {
                totalCost += this._calculateAdjustedCost(prev, customerIndex);
                prev = customerIndex;
            }
            // Add cost of returning to depot
            totalCost += this._calculateAdjustedCost(prev, 0);
        }
        return totalCost; 
    }

    /** Applies pheromone evaporation and deposition based on the best solution of the iteration. */
    _updatePheromones(allSolutions) {
        // 1. Evaporation
        for (let i = 0; i < this.numNodes; i++) {
            for (let j = 0; j < this.numNodes; j++) {
                this.pheromoneMatrix[i][j] *= (1 - this.rho);
            }
        }
        
        // 2. Deposition (using only the best solution found in this iteration)
        const bestIterSolution = allSolutions.reduce((best, current) => current.cost < best.cost ? current : best);
        
        // Calculate the amount of pheromone to deposit 
        const deposit = this.Q / bestIterSolution.cost; 
        
        for (const route of bestIterSolution.routes) {
            if (!route || route.length === 0) continue;
            
            let prev = 0;
            for (const customerIndex of route) {
                this.pheromoneMatrix[prev][customerIndex] += deposit;
                this.pheromoneMatrix[customerIndex][prev] += deposit; // Symmetric
                prev = customerIndex;
            }
            // Deposit pheromone on the arc back to the depot
            this.pheromoneMatrix[prev][0] += deposit;
            this.pheromoneMatrix[0][prev] += deposit;
        }
    }

    /** Executes the main ACO optimization loop. */
    optimize() {
        const startTime = performance.now();
        let iterationHistory = [];

        for (let iter = 0; iter < this.numIterations; iter++) {
            const iterationSolutions = [];
            
            // 1. Ant construction phase
            for (let ant = 0; ant < this.numAnts; ant++) {
                const routes = this._constructAntSolution();
                const cost = this._calculateSolutionCost(routes);
                iterationSolutions.push({ routes, cost });
                
                // Update global best solution
                if (cost < this.bestCost) {
                    this.bestCost = cost;
                    this.bestSolution = routes;
                }
            }
            
            // 2. Pheromone update phase
            this._updatePheromones(iterationSolutions);
            
            iterationHistory.push(this.bestCost);
        }

        const endTime = performance.now();

        return {
            solution: this.bestSolution,
            cost: this.bestCost,
            time: (endTime - startTime) / 1000,
            history: iterationHistory
        };
    }
}


// --- UI, MAP INTEGRATION, AND HANDLERS ---

/** Creates a new customer with random demand and time window */
function createRandomCustomer(lat, lng) {
    const id = nextCustomerId++;
    const demand = 1; // Each customer is 1 delivery point
    
    // Time window logic: standard working hours (8 AM to 5 PM = 8 to 17 in hours)
    const baseHour = Math.floor(Math.random() * 6) + 9; // 9 AM (9) to 3 PM (15)
    let twStart = baseHour + randGaussian(0, 1.5); // Randomize start time slightly
    twStart = Math.max(8, Math.min(16, twStart)); // Keep it within 8 AM to 4 PM
    
    // Time window duration is 2-4 hours
    const twEnd = twStart + Math.floor(Math.random() * 2) + 2; 

    return new Customer(id, lat, lng, demand, twStart, twEnd);
}

/** Main function to run the simulation (called via onclick in index.html) */
window.runSimulation = function() {
    const btn = document.getElementById('runSimulationBtn');
    const numCustomers = customerData.length;

    if (numCustomers === 0) {
        document.getElementById('statusMessage').textContent = 'Error: Please place at least one customer on the map.';
        document.getElementById('statusMessage').classList.add('text-red-600');
        return;
    }

    // 1. Get Selected Vehicle Type
    const vehicleType = document.getElementById('vehicleTypeSelect').value;
    const fleetConfig = {
        [vehicleType]: 1 // Create a fleet of 1 vehicle of the selected type
    };
    
    // UI state update
    btn.disabled = true;
    document.getElementById('statusMessage').textContent = `Running optimization for ${numCustomers} customers...`;
    document.getElementById('statusMessage').classList.add('text-yellow-600');
    document.getElementById('statusMessage').classList.remove('text-gray-500', 'text-green-600', 'text-red-600');

    // 2. Get ACO Parameters from UI
    const numAnts = parseInt(document.getElementById('numAnts').value);
    const numIterations = parseInt(document.getElementById('numIterations').value);
    const alpha = parseFloat(document.getElementById('alpha').value);
    const beta = parseFloat(document.getElementById('beta').value);
    const rho = parseFloat(document.getElementById('rho').value);

    // 3. Setup Simulator
    const params = { numAnts, numIterations, alpha, beta, rho, Q: 1000000 }; 
    currentSimulator = new AcoCvrpSimulator(depotLocation, customerData, fleetConfig, params);
    
    // Run the optimization asynchronously (even with a small delay)
    setTimeout(() => {
        // 4. Run Optimization
        const results = currentSimulator.optimize();

        // 5. Update UI
        updateResults(results);
        drawRoutes(currentSimulator.bestSolution, currentSimulator.customers);

        btn.disabled = false;
        document.getElementById('statusMessage').textContent = 'Optimization Complete!';
        document.getElementById('statusMessage').classList.add('text-green-600');
        document.getElementById('statusMessage').classList.remove('text-yellow-600');

    }, 50); 
}

/** Clears markers, polylines, and resets customer data (called via onclick in index.html) */
window.resetMap = function() {
    // Clear existing Google Maps elements
    markers.forEach(marker => marker.setMap(null));
    polylines.forEach(polyline => polyline.setMap(null));
    markers = [];
    polylines = [];
    
    // Reset simulation data
    customerData = [];
    nextCustomerId = 1;
    currentSimulator = null;
    
    // Re-place depot marker
    placeDepotMarker();

    // Update UI
    document.getElementById('numCustomersDisplay').textContent = '0';
    // Call handleVehicleChange to reset status message correctly
    handleVehicleChange(false); // Pass false to prevent resetting map again
    
    document.getElementById('bestCost').textContent = '-';
    document.getElementById('vehiclesUsed').textContent = '-';
    document.getElementById('routeDetails').textContent = 'No routes calculated yet.';
}


/** Updates the metrics and route details panel */
function updateResults(results) {
    const solution = results.solution;
    const cost = results.cost;
    const vehiclesUsed = solution.filter(r => r.length > 0).length;

    // Cost is in adjusted meters, convert to kilometers for display
    document.getElementById('bestCost').textContent = (cost / 1000).toFixed(2) + ' km';
    document.getElementById('vehiclesUsed').textContent = `${vehiclesUsed} / ${currentSimulator.numVehicles}`;

    // Route Details
    const routeDetailsDiv = document.getElementById('routeDetails');
    if (solution && solution.length > 0) {
        routeDetailsDiv.innerHTML = solution.map((route, index) => {
            if (route.length === 0) return '';
            
            const vehicle = currentSimulator.vehicles[index];
            const vType = vehicle.type.charAt(0).toUpperCase() + vehicle.type.slice(1);
            
            const stopCount = route.length;
            const routeStr = route.map(cIdx => `C${cIdx}`).join(' → ');
            const totalRouteCost = (currentSimulator._calculateSolutionCost([route]) / 1000).toFixed(2);

            // --- Calculate Total Route Time ---
            let routeTime = 0;
            let lastNode = 0; // Start at depot
            for (const cIdx of route) {
                const customer = customerData[cIdx - 1];
                const travelTime = currentSimulator._calculateTravelTime(lastNode, cIdx, vehicle);
                const arrivalTime = routeTime + travelTime;
                
                // Wait if arriving early for time window
                const serviceStart = Math.max(arrivalTime, customer.timeWindowStart);
                routeTime = serviceStart + customer.serviceTime;
                lastNode = cIdx;
            }
            // Add time to return to depot
            const travelTimeBack = currentSimulator._calculateTravelTime(lastNode, 0, vehicle);
            routeTime += travelTimeBack;
            const totalTimeHours = routeTime.toFixed(2);
            // --- End Time Calculation ---

            return `
                <p class="mb-1">
                    <span class="text-indigo-600">V${vehicle.id + 1} (${vType})</span> 
                    (Stops: ${stopCount}/${vehicle.maxStops}, 
                     Time: ${totalTimeHours}h, 
                     Cost: ${totalRouteCost} km): 
                    Depot → ${routeStr} → Depot
                </p>
            `;
        }).join('');
    } else {
        routeDetailsDiv.textContent = 'Failed to find a feasible solution or no routes generated.';
    }
}

/** Initializes the Google Map instance (called by Google Maps API script in index.html) */
window.initMap = function() {
    map = new google.maps.Map(document.getElementById("map"), {
        center: depotLocation,
        zoom: INITIAL_MAP_ZOOM,
        mapTypeId: 'roadmap',
        streetViewControl: false,
        mapTypeControl: false,
        fullscreenControl: false
    });

    // Place initial depot marker
    placeDepotMarker();

    // Add click listener to place customers
    map.addListener('click', handleMapClick);

    // Add listener for vehicle dropdown
    document.getElementById('vehicleTypeSelect').addEventListener('change', () => handleVehicleChange(true));
    // Set initial state from the dropdown
    handleVehicleChange(false);
}

/** * Places the fixed Depot marker (Index 0 in the location array)
 * === MODIFIED: Changed icon to a visible yellow "home" icon ===
 */
function placeDepotMarker() {
    const marker = new google.maps.Marker({
        position: depotLocation,
        map: map,
        icon: {
            // Swapped the blue shield for a yellow home icon for better visibility
            url: "data:image/svg+xml;charset=UTF-8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%23eab308'><path d='M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8h5z'/></svg>",
            scaledSize: new google.maps.Size(32, 32),
            anchor: new google.maps.Point(16, 16)
        },
        title: 'Depot (HUB) - Barangay San Manuel, Puerto Princesa'
    });
    markers.push(marker);
}

/** Handles click event to place a new customer marker */
function handleMapClick(mapsMouseEvent) {
    const statusMessage = document.getElementById('statusMessage');

    // Check if customer limit is reached
    if (customerData.length >= maxCustomers) {
        statusMessage.textContent = `Error: Cannot add more customers. Limit for ${selectedVehicleType} is ${maxCustomers}.`;
        statusMessage.classList.add('text-red-600');
        statusMessage.classList.remove('text-gray-500', 'text-yellow-600', 'text-green-600');
        return;
    }

    const lat = mapsMouseEvent.latLng.lat();
    const lng = mapsMouseEvent.latLng.lng();

    // Create and add the new customer
    const newCustomer = createRandomCustomer(lat, lng);
    customerData.push(newCustomer);
    
    // Create a marker for the customer
    const marker = new google.maps.Marker({
        position: { lat, lng },
        map: map,
        label: {
            text: `C${newCustomer.id}`,
            color: "white",
            fontWeight: "bold"
        },
        icon: {
            path: google.maps.SymbolPath.CIRCLE,
            scale: 8,
            fillColor: colorPalette[newCustomer.id % colorPalette.length], 
            fillOpacity: 0.8,
            strokeWeight: 0
        },
        title: `C${newCustomer.id} | TW:${newCustomer.timeWindowStart.toFixed(1)}-${newCustomer.timeWindowEnd.toFixed(1)}`
    });
    markers.push(marker);

    // Update UI
    document.getElementById('numCustomersDisplay').textContent = customerData.length;
    statusMessage.textContent = `Customer C${newCustomer.id} placed. (${customerData.length}/${maxCustomers} stops filled).`;
    statusMessage.classList.remove('text-red-600', 'text-yellow-600', 'text-green-600');
    statusMessage.classList.add('text-gray-500');
}


/** * NEW: Handles change event from the vehicle selection dropdown
 * Updates global state and resets map if capacity is exceeded.
 */
function handleVehicleChange(shouldResetMapIfConflict = false) {
    const select = document.getElementById('vehicleTypeSelect');
    const selectedOption = select.options[select.selectedIndex];
    
    selectedVehicleType = selectedOption.value;
    const newMaxCustomers = parseInt(selectedOption.getAttribute('data-capacity'));
    const vType = selectedOption.textContent;
    const statusMessage = document.getElementById('statusMessage');

    if (shouldResetMapIfConflict && customerData.length > newMaxCustomers) {
        // If user plots 12 customers (Tricycle) then switches to Motorcycle (10),
        // we must reset the map.
        resetMap();
        statusMessage.textContent = `Map reset. New vehicle selected: ${vType}.`;
    } else {
        // Otherwise, just update the state
        statusMessage.textContent = `Vehicle set to: ${vType}. Max ${newMaxCustomers} stops.`;
    }

    maxCustomers = newMaxCustomers;
    statusMessage.classList.remove('text-red-600', 'text-yellow-600', 'text-green-600');
    statusMessage.classList.add('text-gray-500');
}


/** Draws the routes on the Google Map using Polylines */
function drawRoutes(routes, customers) {
    // Clear existing polylines
    polylines.forEach(polyline => polyline.setMap(null));
    polylines = [];
    
    if (!routes || routes.length === 0) return;

    routes.forEach((route, vIndex) => {
        if (!route || route.length === 0) return;
        
        const color = colorPalette[vIndex % colorPalette.length];
        let path = [];
        
        // Start at Depot
        path.push(depotLocation);

        route.forEach(cIdx => {
            const customer = customers[cIdx - 1]; 
            path.push({ lat: customer.lat, lng: customer.lng });
        });

        // Return to Depot
        path.push(depotLocation);

        const polyline = new google.maps.Polyline({
            path: path,
            geodesic: true, 
            strokeColor: color,
            strokeOpacity: 0.7,
            strokeWeight: 3.5,
            icons: [{
                // Add an arrow icon to indicate route direction
                icon: { path: google.maps.SymbolPath.FORWARD_CLOSED_ARROW, scale: 3, strokeColor: color, fillOpacity: 1, fillColor: color },
                offset: '100%'
            }]
        });

        polyline.setMap(map);
        polylines.push(polyline);
    });
}

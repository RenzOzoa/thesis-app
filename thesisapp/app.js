// === CONFIGURATION AND GLOBAL STATE ===
// Fixed depot location: Barangay San Manuel, Puerto Princesa, Philippines
const depotLocation = { lat: 9.7735, lng: 118.7524 }; 
const INITIAL_MAP_ZOOM = 13;

// --- Heterogeneous Fleet Properties ---
const VEHICLE_PROPERTIES = {
    motorcycle: { speed: 40, capacity: 10 },
    tricycle: { speed: 35, capacity: 15 },
    car: { speed: 40, capacity: 20 },
    van: { speed: 40, capacity: 30 }
};

// Global State
let map;
let markers = [];
let directionRenderers = []; // Replaces polylines
let currentSimulator = null;
let customerData = [];
let nextCustomerId = 1;
let selectedVehicleType = 'motorcycle';
let maxCustomers = 10;

// --- NEW: Google Maps Services ---
let directionsService;
let distanceMatrixService;

const colorPalette = [
    '#ef4444', '#f97316', '#eab308', '#22c55e', 
    '#06b6d4', '#3b82f6', '#8b5cf6', '#d946ef',
    '#64748b', '#dc2626', '#34d399', '#f472b6' 
];


// --- CORE DATA STRUCTURES (Models) ---

class Customer {
    /**
     * MODIFIED: serviceTime default is now 8 minutes (8 / 60 hours)
     */
    constructor(id, lat, lng, demand, timeWindowStart, timeWindowEnd, serviceTime = (8 / 60)) {
        this.id = id;
        this.lat = lat;
        this.lng = lng;
        this.demand = demand; // 1 delivery point
        this.timeWindowStart = timeWindowStart; // (No longer used in calculations, but kept in data)
        this.timeWindowEnd = timeWindowEnd;     // (No longer used in calculations, but kept in data)
        this.serviceTime = serviceTime;         // in hours
    }
}

class Vehicle {
    constructor(id, type, speed, maxStops) {
        this.id = id;
        this.type = type;
        this.speed = speed; // km/h
        this.maxStops = maxStops;
        this.currentTime = 0; // in hours
        this.route = [];
        this.currentLocationIndex = 0;
    }

    reset() {
        this.currentTime = 0;
        this.route = [];
        this.currentLocationIndex = 0;
    }
}


// --- UTILITY FUNCTIONS ---

/** Generates a random number following a Gaussian (Normal) distribution. */
const randGaussian = (mean = 0, stdDev = 1) => {
    let u = 0, v = 0;
    while (u === 0) u = Math.random(); 
    while (v === 0) v = Math.random();
    let z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return z * stdDev + mean;
};

/**
 * --- NEW: Formats decimal hours into a "X hours Y minutes" string ---
 */
function formatDecimalHours(decimalHours) {
    const totalMinutes = Math.round(decimalHours * 60);
    
    if (totalMinutes === 0) {
        return '0 minutes';
    }

    const hours = Math.floor(totalMinutes / 60);
    const minutes = totalMinutes % 60;

    let parts = [];
    if (hours > 0) {
        parts.push(`${hours} hour${hours > 1 ? 's' : ''}`);
    }
    if (minutes > 0) {
        parts.push(`${minutes} minute${minutes !== 1 ? 's' : ''}`);
    }

    return parts.join(' ');
}


// --- ACO CVRP CORE LOGIC (Solver) ---

class AcoCvrpSimulator {
    /**
     * MODIFIED: Accepts matrixData from the Google Distance Matrix API.
     */
    constructor(depotLocation, customers, fleetConfig, params, matrixData) {
        this.depotLocation = depotLocation;
        this.customers = customers;
        
        // Build vehicle fleet
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

        // ACO Parameters
        this.numAnts = params.numAnts;
        this.numIterations = params.numIterations;
        this.alpha = params.alpha;
        this.beta = params.beta;
        this.rho = params.rho;
        this.Q = params.Q;
        this.tau0 = params.tau0 || 0.1;
        
        // Locations (0: Depot, 1..N: Customers)
        this.locations = [depotLocation].concat(this.customers.map(c => ({ lat: c.lat, lng: c.lng })));
        this.numNodes = this.locations.length;
        
        // --- NEW: Use pre-fetched Google Maps data ---
        this.distanceMatrix = matrixData.distances; // in meters
        this.timeMatrix = matrixData.times;         // in seconds
        
        // Initialize Pheromone Matrix
        this.pheromoneMatrix = Array(this.numNodes).fill(0).map(() => 
            Array(this.numNodes).fill(this.tau0)
        );
        
        this.bestSolution = null;
        this.bestCost = Infinity;
    }

    /**
     * MODIFIED: Calculates travel time in hours from pre-fetched Google data.
     */
    _calculateTravelTime(i, j, vehicle) {
        const baseTimeSeconds = this.timeMatrix[i][j];
        if (baseTimeSeconds === Infinity) {
            return Infinity;
        }
        // Convert real travel time from seconds to hours
        return baseTimeSeconds / 3600.0;
    }
    
    /**
     * MODIFIED: Calculates cost (distance in meters) from pre-fetched Google data.
     */
    _calculateAdjustedCost(i, j) {
        // Cost is the real road distance in meters
        return this.distanceMatrix[i][j];
    }

    /** Calculates the heuristic value ($\eta_{ij}$), which is the inverse of the adjusted cost. */
    _calculateHeuristic(i, j) {
        const adjustedCost = this._calculateAdjustedCost(i, j);
        if (adjustedCost === 0 || adjustedCost === Infinity) {
            return 0; // Avoid division by zero or using bad paths
        }
        return 1000000.0 / adjustedCost; // Heuristic is inverse of distance
    }

    /**
     * MODIFIED: Calculates probabilities.
     * REMOVED the Time Window (twFactor) logic.
     */
    _calculateProbability(current, unvisited, vehicle) {
        const probabilities = {};
        let total = 0;

        for (const nextCustomerIndex of unvisited) {
            // 1. Capacity Constraint
            if (vehicle.route.length >= vehicle.maxStops) {
                continue;
            }
            
            // 2. Check for unreachable path
            const travelTime = this._calculateTravelTime(current, nextCustomerIndex, vehicle);
            if (travelTime === Infinity) {
                continue; // Unreachable path
            }
            
            // 3. Probability Calculation (REMOVED twFactor)
            const pheromone = Math.pow(this.pheromoneMatrix[current][nextCustomerIndex], this.alpha);
            const heuristic = Math.pow(this._calculateHeuristic(current, nextCustomerIndex), this.beta);
            
            const probability = pheromone * heuristic;
            
            if (probability > 0) {
                probabilities[nextCustomerIndex] = probability;
                total += probability;
            }
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
        // Fallback for floating point errors
        return parseInt(Object.keys(probabilities).pop());
    }

    /** * An ant constructs a full set of routes for all available vehicles.
     * MODIFIED: REMOVED "waiting time" for time windows.
     */
    _constructAntSolution() {
        let unvisited = new Set(Array.from({ length: this.numNodes - 1 }, (_, i) => i + 1));
        const routes = [];

        this.vehicles.forEach(v => v.reset());
        
        for (const vehicle of this.vehicles) {
            if (unvisited.size === 0) break;
            
            let current = 0; // Start at depot (index 0)
            
            while (unvisited.size > 0) {
                const probabilities = this._calculateProbability(current, Array.from(unvisited), vehicle);
                
                if (Object.keys(probabilities).length === 0) {
                    break; // No valid moves left
                }
                
                const nextCustomerIndex = this._selectNextCustomer(probabilities);
                
                if (nextCustomerIndex === -1) {
                    break; 
                }
                
                const customer = this.customers[nextCustomerIndex - 1];
                
                vehicle.route.push(nextCustomerIndex);
                
                const travelTime = this._calculateTravelTime(current, nextCustomerIndex, vehicle);
                const arrivalTime = vehicle.currentTime + travelTime;
                
                // --- MODIFICATION ---
                // REMOVED: const serviceStart = Math.max(arrivalTime, customer.timeWindowStart);
                // The ant no longer waits.
                vehicle.currentTime = arrivalTime + customer.serviceTime;
                // --- END MODIFICATION ---
                
                unvisited.delete(nextCustomerIndex);
                current = nextCustomerIndex;
            }

            // Close route: return to depot (0)
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

    /** Applies pheromone evaporation and deposition. */
    _updatePheromones(allSolutions) {
        // 1. Evaporation
        for (let i = 0; i < this.numNodes; i++) {
            for (let j = 0; j < this.numNodes; j++) {
                this.pheromoneMatrix[i][j] *= (1 - this.rho);
            }
        }
        
        // 2. Deposition (using only the best solution found in this iteration)
        const bestIterSolution = allSolutions.reduce((best, current) => current.cost < best.cost ? current : best);
        
        const deposit = this.Q / bestIterSolution.cost; 
        
        for (const route of bestIterSolution.routes) {
            if (!route || route.length === 0) continue;
            
            let prev = 0;
            for (const customerIndex of route) {
                this.pheromoneMatrix[prev][customerIndex] += deposit;
                this.pheromoneMatrix[customerIndex][prev] += deposit; // Symmetric
                prev = customerIndex;
            }
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
            
            for (let ant = 0; ant < this.numAnts; ant++) {
                const routes = this._constructAntSolution();
                const cost = this._calculateSolutionCost(routes);
                iterationSolutions.push({ routes, cost });
                
                if (cost < this.bestCost) {
                    this.bestCost = cost;
                    this.bestSolution = routes;
                }
            }
            
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

/** * Creates a new customer with random demand and time window
 * MODIFIED: The serviceTime is now pulled from the Customer class default (8 mins)
 */
function createRandomCustomer(lat, lng) {
    const id = nextCustomerId++;
    const demand = 1; // Each customer is 1 delivery point
    
    // Time windows are still generated for the data model, but not used in time calculations
    const baseHour = Math.floor(Math.random() * 6) + 9; // 9 AM (9) to 3 PM (15)
    let twStart = baseHour + randGaussian(0, 1.5);
    twStart = Math.max(8, Math.min(16, twStart)); // 8 AM to 4 PM
    const twEnd = twStart + Math.floor(Math.random() * 2) + 2; // 2-4 hour window

    // The 'serviceTime' parameter is no longer passed, so it uses the new default
    return new Customer(id, lat, lng, demand, twStart, twEnd);
}

/**
 * --- NEW: Step 1 - Fetch Distance and Time Matrices from Google ---
 */
function fetchDistanceMatrix(locations) {
    const statusMessage = document.getElementById('statusMessage');
    statusMessage.textContent = 'Fetching road network data from Google...';
    statusMessage.classList.add('text-yellow-600');
    statusMessage.classList.remove('text-gray-500');
    
    return new Promise((resolve) => {
        const request = {
            origins: locations,
            destinations: locations,
            travelMode: google.maps.TravelMode.DRIVING,
            unitSystem: google.maps.UnitSystem.METRIC,
        };

        distanceMatrixService.getDistanceMatrix(request, (response, status) => {
            if (status !== 'OK') {
                console.error('Distance Matrix Error:', status);
                statusMessage.textContent = `Error fetching road data: ${status}. Using straight-line distance as fallback.`;
                statusMessage.classList.add('text-red-600');
                resolve(null); // Indicate failure
                return;
            }

            const distanceMatrix = [];
            const timeMatrix = [];

            response.rows.forEach((row, i) => {
                distanceMatrix[i] = [];
                timeMatrix[i] = [];
                row.elements.forEach((element, j) => {
                    if (element.status === 'OK') {
                        distanceMatrix[i][j] = element.distance.value; // meters
                        timeMatrix[i][j] = element.duration.value;     // seconds
                    } else {
                        // Handle unreachable locations
                        distanceMatrix[i][j] = Infinity;
                        timeMatrix[i][j] = Infinity;
                    }
                });
            });
            
            resolve({ distances: distanceMatrix, times: timeMatrix });
        });
    });
}


/**
 * MODIFIED: Main function to run the simulation. Now ASYNC.
 */
window.runSimulation = async function() {
    const btn = document.getElementById('runSimulationBtn');
    const statusMessage = document.getElementById('statusMessage');
    const numCustomers = customerData.length;

    if (numCustomers === 0) {
        statusMessage.textContent = 'Error: Please place at least one customer on the map.';
        statusMessage.classList.add('text-red-600');
        return;
    }

    // --- Start UI Loading State ---
    btn.disabled = true;
    statusMessage.textContent = 'Starting simulation...';
    statusMessage.classList.add('text-yellow-600');
    statusMessage.classList.remove('text-gray-500', 'text-green-600', 'text-red-600');
    
    try {
        // --- STEP 1: Fetch Real Road Data ---
        const locations = [depotLocation, ...customerData.map(c => ({ lat: c.lat, lng: c.lng }))];
        const matrixData = await fetchDistanceMatrix(locations);

        if (!matrixData) {
            // Error was already handled by fetchDistanceMatrix
            btn.disabled = false;
            return;
        }
        
        statusMessage.textContent = `Running optimization for ${numCustomers} customers...`;
        
        // --- STEP 2: Run ACO with Real Data ---
        const vehicleType = document.getElementById('vehicleTypeSelect').value;
        const fleetConfig = { [vehicleType]: 1 };
        
        const numAnts = parseInt(document.getElementById('numAnts').value);
        const numIterations = parseInt(document.getElementById('numIterations').value);
        const alpha = parseFloat(document.getElementById('alpha').value);
        const beta = parseFloat(document.getElementById('beta').value);
        const rho = parseFloat(document.getElementById('rho').value);

        const params = { numAnts, numIterations, alpha, beta, rho, Q: 1000000 }; 
        
        // Pass the fetched matrixData to the simulator
        currentSimulator = new AcoCvrpSimulator(depotLocation, customerData, fleetConfig, params, matrixData);
        
        // Run optimization (this is fast, the API call was the slow part)
        const results = currentSimulator.optimize();

        // --- STEP 3 & 4: Update UI and Draw Routes ---
        updateResults(results); // Step 4 (Now uses real time)
        drawRoutes(currentSimulator.bestSolution, currentSimulator.customers); // Step 3

        // --- Finish UI State ---
        btn.disabled = false;
        statusMessage.textContent = 'Optimization Complete!';
        statusMessage.classList.add('text-green-600');
        statusMessage.classList.remove('text-yellow-600');

    } catch (error) {
        console.error("Simulation failed:", error);
        statusMessage.textContent = `Simulation failed: ${error.message}`;
        statusMessage.classList.add('text-red-600');
        btn.disabled = false;
    }
}

/** Clears markers, routes, and resets data */
window.resetMap = function() {
    // Clear existing Google Maps elements
    markers.forEach(marker => marker.setMap(null));
    directionRenderers.forEach(renderer => renderer.setMap(null)); // Clear routes
    markers = [];
    directionRenderers = [];
    
    // Reset simulation data
    customerData = [];
    nextCustomerId = 1;
    currentSimulator = null;
    
    // Re-place depot marker
    placeDepotMarker();

    // Update UI
    document.getElementById('numCustomersDisplay').textContent = '0';
    handleVehicleChange(false); // Resets status message
    
    document.getElementById('bestCost').textContent = '-';
    document.getElementById('vehiclesUsed').textContent = '-';
    document.getElementById('routeDetails').textContent = 'No routes calculated yet.';
}


/** * Updates the metrics and route details panel.
 * MODIFIED: Uses new formatDecimalHours() for time display.
 */
function updateResults(results) {
    const solution = results.solution;
    const cost = results.cost; // This is now real road distance (meters)
    const vehiclesUsed = solution.filter(r => r.length > 0).length;

    document.getElementById('bestCost').textContent = (cost / 1000).toFixed(2) + ' km';
    document.getElementById('vehiclesUsed').textContent = `${vehiclesUsed} / ${currentSimulator.numVehicles}`;

    const routeDetailsDiv = document.getElementById('routeDetails');
    if (solution && solution.length > 0) {
        routeDetailsDiv.innerHTML = solution.map((route, index) => {
            if (route.length === 0) return '';
            
            const vehicle = currentSimulator.vehicles[index];
            const vType = vehicle.type.charAt(0).toUpperCase() + vehicle.type.slice(1);
            
            const stopCount = route.length;
            const routeStr = route.map(cIdx => `C${cIdx}`).join(' → ');
            const totalRouteCost = (currentSimulator._calculateSolutionCost([route]) / 1000).toFixed(2);

            // --- MODIFIED: Calculate Total Route Time (No waiting) ---
            let routeTime = 0; // in hours
            let lastNode = 0; // Depot
            for (const cIdx of route) {
                const customer = customerData[cIdx - 1];
                const travelTime = currentSimulator._calculateTravelTime(lastNode, cIdx, vehicle);
                const arrivalTime = routeTime + travelTime;
                
                routeTime = arrivalTime + customer.serviceTime; // No waiting
                lastNode = cIdx;
            }
            const travelTimeBack = currentSimulator._calculateTravelTime(lastNode, 0, vehicle);
            routeTime += travelTimeBack;
            
            // --- NEW: Use the time formatter ---
            const totalTimeString = formatDecimalHours(routeTime);
            // --- END NEW ---

            return `
                <p class="mb-1">
                    <span class="text-indigo-600">V${vehicle.id + 1} (${vType})</span> 
                    (Stops: ${stopCount}/${vehicle.maxStops}, 
                     Time: ${totalTimeString}, 
                     Cost: ${totalRouteCost} km): 
                    Depot → ${routeStr} → Depot
                </p>
            `;
        }).join('');
    } else {
        routeDetailsDiv.textContent = 'Failed to find a feasible solution or no routes generated.';
    }
}

/** Initializes the Google Map instance */
window.initMap = function() {
    map = new google.maps.Map(document.getElementById("map"), {
        center: depotLocation,
        zoom: INITIAL_MAP_ZOOM,
        mapTypeId: 'roadmap',
        streetViewControl: false,
        mapTypeControl: false,
        fullscreenControl: false
    });
    
    // --- NEW: Initialize Google services ---
    directionsService = new google.maps.DirectionsService();
    distanceMatrixService = new google.maps.DistanceMatrixService();

    // Place initial depot marker
    placeDepotMarker();

    // Add click listener to place customers
    map.addListener('click', handleMapClick);

    // Add listener for vehicle dropdown
    document.getElementById('vehicleTypeSelect').addEventListener('change', () => handleVehicleChange(true));
    handleVehicleChange(false);
}

/** Places the fixed Depot marker (Yellow Home) */
function placeDepotMarker() {
    const marker = new google.maps.Marker({
        position: depotLocation,
        map: map,
        icon: {
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
    const newCustomer = createRandomCustomer(lat, lng);
    customerData.push(newCustomer);
    
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
        // Title no longer shows time window as it's not used
        title: `C${newCustomer.id}`
    });
    markers.push(marker);

    // Update UI
    document.getElementById('numCustomersDisplay').textContent = customerData.length;
    statusMessage.textContent = `Customer C${newCustomer.id} placed. (${customerData.length}/${maxCustomers} stops filled).`;
    statusMessage.classList.remove('text-red-600', 'text-yellow-600', 'text-green-600');
    statusMessage.classList.add('text-gray-500');
}


/** Handles change event from the vehicle selection dropdown */
function handleVehicleChange(shouldResetMapIfConflict = false) {
    const select = document.getElementById('vehicleTypeSelect');
    const selectedOption = select.options[select.selectedIndex];
    
    selectedVehicleType = selectedOption.value;
    const newMaxCustomers = parseInt(selectedOption.getAttribute('data-capacity'));
    const vType = selectedOption.textContent;
    const statusMessage = document.getElementById('statusMessage');

    if (shouldResetMapIfConflict && customerData.length > newMaxCustomers) {
        resetMap();
        statusMessage.textContent = `Map reset. New vehicle selected: ${vType}.`;
    } else {
        statusMessage.textContent = `Vehicle set to: ${vType}. Max ${newMaxCustomers} stops.`;
    }

    maxCustomers = newMaxCustomers;
    statusMessage.classList.remove('text-red-600', 'text-yellow-600', 'text-green-600');
    statusMessage.classList.add('text-gray-500');
}


/** * --- STEP 3: Visualize the Real Route ---
 * MODIFIED: Draws the routes on the map using DirectionsService
 * This will draw road-snapped routes.
 */
function drawRoutes(routes, customers) {
    // Clear existing route renderers
    directionRenderers.forEach(renderer => renderer.setMap(null));
    directionRenderers = [];
    
    if (!routes || routes.length === 0) return;

    routes.forEach((route, vIndex) => {
        if (!route || route.length === 0) return;
        
        const color = colorPalette[vIndex % colorPalette.length];
        
        // Build the list of stops for the Directions API
        const routeStops = [];
        routeStops.push({ location: depotLocation, stopover: true });
        route.forEach(cIdx => {
            const customer = customers[cIdx - 1]; 
            routeStops.push({ location: { lat: customer.lat, lng: customer.lng }, stopover: true });
        });
        
        if (routeStops.length < 2) return; // Not a valid route

        const origin = routeStops[0].location;
        const destination = depotLocation; // Always return to depot
        const waypoints = routeStops.slice(1); // All customers are waypoints

        const request = {
            origin: origin,
            destination: destination,
            waypoints: waypoints,
            travelMode: google.maps.TravelMode.DRIVING,
            optimizeWaypoints: false // IMPORTANT: We use ACO's order, not Google's
        };

        // Call the Directions Service
        directionsService.route(request, (response, status) => {
            if (status === 'OK') {
                // Create a new renderer for this specific route
                const renderer = new google.maps.DirectionsRenderer({
                    map: map,
                    directions: response,
                    suppressMarkers: true, // Hide A, B, C default markers
                    polylineOptions: {
                        strokeColor: color,
                        strokeOpacity: 0.8,
                        strokeWeight: 5
                    }
                });
                directionRenderers.push(renderer);
            } else {
                console.error('Directions request failed due to ' + status);
            }
        });
    });
}

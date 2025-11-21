// === CONFIGURATION AND GLOBAL STATE ===
// Fixed depot location: Flash Express, Puerto Princesa, Philippines
const depotLocation = { lat: 9.7408, lng: 118.7316 }; 
const INITIAL_MAP_ZOOM = 13;

// --- Heterogeneous Fleet Properties ---
const VEHICLE_PROPERTIES = {
    motorcycle: { speed: 40, capacity: 20 },
    tricycle: { speed: 35, capacity: 25 },
    car: { speed: 40, capacity: 30 },
    van: { speed: 40, capacity: 50 } 
};

// Global State
let map;
let markers = [];
let directionRenderers = []; // Replaces polylines
let currentSimulator = null;
let customerData = [];
let nextCustomerId = 1;
let selectedVehicleType = 'motorcycle';
let maxCustomers = 50; 

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
    // Service time default: 5 minutes (5/60 hours)
    constructor(id, lat, lng, demand, timeWindowStart, timeWindowEnd, serviceTime = (5 / 60)) {
        this.id = id;
        this.lat = lat;
        this.lng = lng;
        this.demand = demand; 
        this.timeWindowStart = timeWindowStart; 
        this.timeWindowEnd = timeWindowEnd;     
        this.serviceTime = serviceTime;         
    }
}

class Vehicle {
    constructor(id, type, speed, maxStops) {
        this.id = id;
        this.type = type;
        this.speed = speed; 
        this.maxStops = maxStops;
        this.currentTime = 0; 
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

const randGaussian = (mean = 0, stdDev = 1) => {
    let u = 0, v = 0;
    while (u === 0) u = Math.random(); 
    while (v === 0) v = Math.random();
    let z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return z * stdDev + mean;
};

function formatDecimalHours(decimalHours) {
    const totalMinutes = Math.round(decimalHours * 60);
    if (totalMinutes === 0) return '0 minutes';
    const hours = Math.floor(totalMinutes / 60);
    const minutes = totalMinutes % 60;
    let parts = [];
    if (hours > 0) parts.push(`${hours} hour${hours > 1 ? 's' : ''}`);
    if (minutes > 0) parts.push(`${minutes} minute${minutes !== 1 ? 's' : ''}`);
    return parts.join(' ');
}


// --- ACO CVRP CORE LOGIC (Solver) ---

class AcoCvrpSimulator {
    constructor(depotLocation, customers, fleetConfig, params, matrixData) {
        this.depotLocation = depotLocation;
        this.customers = customers;
        
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

        this.numAnts = params.numAnts;
        this.numIterations = params.numIterations;
        this.alpha = params.alpha;
        this.beta = params.beta;
        this.rho = params.rho;
        this.Q = params.Q;
        this.tau0 = params.tau0 || 0.1;
        
        this.locations = [depotLocation].concat(this.customers.map(c => ({ lat: c.lat, lng: c.lng })));
        this.numNodes = this.locations.length;
        
        this.distanceMatrix = matrixData.distances; 
        this.timeMatrix = matrixData.times;         
        
        this.pheromoneMatrix = Array(this.numNodes).fill(0).map(() => 
            Array(this.numNodes).fill(this.tau0)
        );
        
        this.bestSolution = null;
        this.bestCost = Infinity;
        this.convergenceIteration = 0;
    }

    _calculateTravelTime(i, j, vehicle) {
        const baseTimeSeconds = this.timeMatrix[i][j];
        if (baseTimeSeconds === Infinity) return Infinity;
        return baseTimeSeconds / 3600.0;
    }
    
    _calculateAdjustedCost(i, j) {
        return this.distanceMatrix[i][j];
    }

    _calculateHeuristic(i, j) {
        const adjustedCost = this._calculateAdjustedCost(i, j);
        if (adjustedCost === 0 || adjustedCost === Infinity) return 0; 
        return 1000000.0 / adjustedCost; 
    }

    _calculateProbability(current, unvisited, vehicle) {
        const probabilities = {};
        let total = 0;

        for (const nextCustomerIndex of unvisited) {
            if (vehicle.route.length >= vehicle.maxStops) continue;
            
            const travelTime = this._calculateTravelTime(current, nextCustomerIndex, vehicle);
            if (travelTime === Infinity) continue; 
            
            const pheromone = Math.pow(this.pheromoneMatrix[current][nextCustomerIndex], this.alpha);
            const heuristic = Math.pow(this._calculateHeuristic(current, nextCustomerIndex), this.beta);
            
            const probability = pheromone * heuristic;
            
            if (probability > 0) {
                probabilities[nextCustomerIndex] = probability;
                total += probability;
            }
        }

        if (total > 0) {
            for (const customerIndex in probabilities) {
                probabilities[customerIndex] /= total;
            }
        }
        return probabilities;
    }

    _selectNextCustomer(probabilities) {
        if (Object.keys(probabilities).length === 0) return -1;
        const rand = Math.random();
        let cumulative = 0;
        for (const customerIndexStr in probabilities) {
            const customerIndex = parseInt(customerIndexStr);
            cumulative += probabilities[customerIndexStr];
            if (rand <= cumulative) return customerIndex;
        }
        return parseInt(Object.keys(probabilities).pop());
    }

    _constructAntSolution() {
        let unvisited = new Set(Array.from({ length: this.numNodes - 1 }, (_, i) => i + 1));
        const routes = [];

        this.vehicles.forEach(v => v.reset());
        
        for (const vehicle of this.vehicles) {
            if (unvisited.size === 0) break;
            let current = 0; 
            
            while (unvisited.size > 0) {
                const probabilities = this._calculateProbability(current, Array.from(unvisited), vehicle);
                if (Object.keys(probabilities).length === 0) break; 
                
                const nextCustomerIndex = this._selectNextCustomer(probabilities);
                if (nextCustomerIndex === -1) break; 
                
                const customer = this.customers[nextCustomerIndex - 1];
                vehicle.route.push(nextCustomerIndex);
                
                const travelTime = this._calculateTravelTime(current, nextCustomerIndex, vehicle);
                const arrivalTime = vehicle.currentTime + travelTime;
                vehicle.currentTime = arrivalTime + customer.serviceTime;
                
                unvisited.delete(nextCustomerIndex);
                current = nextCustomerIndex;
            }
            if (vehicle.route.length > 0) {
                const travelTimeBack = this._calculateTravelTime(current, 0, vehicle);
                vehicle.currentTime += travelTimeBack;
            }
            routes.push(vehicle.route);
        }
        return routes;
    }
    
    _calculateSolutionCost(routes) {
        let totalCost = 0;
        for (const route of routes) {
            if (!route || route.length === 0) continue;
            let prev = 0; 
            for (const customerIndex of route) {
                totalCost += this._calculateAdjustedCost(prev, customerIndex);
                prev = customerIndex;
            }
            totalCost += this._calculateAdjustedCost(prev, 0);
        }
        return totalCost; 
    }

    _updatePheromones(allSolutions) {
        for (let i = 0; i < this.numNodes; i++) {
            for (let j = 0; j < this.numNodes; j++) {
                this.pheromoneMatrix[i][j] *= (1 - this.rho);
            }
        }
        const bestIterSolution = allSolutions.reduce((best, current) => current.cost < best.cost ? current : best);
        const deposit = this.Q / bestIterSolution.cost; 
        
        for (const route of bestIterSolution.routes) {
            if (!route || route.length === 0) continue;
            let prev = 0;
            for (const customerIndex of route) {
                this.pheromoneMatrix[prev][customerIndex] += deposit;
                this.pheromoneMatrix[customerIndex][prev] += deposit; 
                prev = customerIndex;
            }
            this.pheromoneMatrix[prev][0] += deposit;
            this.pheromoneMatrix[0][prev] += deposit;
        }
    }

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
                    this.convergenceIteration = iter + 1; 
                }
            }
            this._updatePheromones(iterationSolutions);
            iterationHistory.push(this.bestCost);
        }

        const endTime = performance.now();
        return {
            solution: this.bestSolution,
            cost: this.bestCost,
            convergenceIteration: this.convergenceIteration, 
            time: (endTime - startTime) / 1000,
            history: iterationHistory
        };
    }
}


// --- UI, MAP INTEGRATION, AND HANDLERS ---

function createRandomCustomer(lat, lng) {
    const id = nextCustomerId++;
    const demand = 1; 
    const baseHour = Math.floor(Math.random() * 6) + 9; 
    let twStart = baseHour + randGaussian(0, 1.5);
    twStart = Math.max(8, Math.min(16, twStart)); 
    const twEnd = twStart + Math.floor(Math.random() * 2) + 2; 
    return new Customer(id, lat, lng, demand, twStart, twEnd);
}

async function fetchDistanceMatrix(locations) {
    const statusMessage = document.getElementById('statusMessage');
    statusMessage.textContent = 'Step 1/3: Fetching road network data...';
    statusMessage.classList.add('text-yellow-600');
    statusMessage.classList.remove('text-gray-500');
    
    const n = locations.length;
    const distanceMatrix = Array(n).fill(0).map(() => Array(n).fill(Infinity));
    const timeMatrix = Array(n).fill(0).map(() => Array(n).fill(Infinity));
    const CHUNK_SIZE = 10; 
    const tasks = [];
    for (let i = 0; i < n; i += CHUNK_SIZE) {
        for (let j = 0; j < n; j += CHUNK_SIZE) {
            tasks.push({ rowStart: i, colStart: j });
        }
    }

    const processChunk = (task) => {
        return new Promise((resolve) => {
            const origins = locations.slice(task.rowStart, task.rowStart + CHUNK_SIZE);
            const destinations = locations.slice(task.colStart, task.colStart + CHUNK_SIZE);
            const request = {
                origins: origins,
                destinations: destinations,
                travelMode: google.maps.TravelMode.DRIVING,
                unitSystem: google.maps.UnitSystem.METRIC,
            };
            distanceMatrixService.getDistanceMatrix(request, (response, status) => {
                if (status !== 'OK') {
                    console.error('Distance Matrix Chunk Error:', status);
                    resolve(); 
                    return;
                }
                response.rows.forEach((row, rIdx) => {
                    row.elements.forEach((element, cIdx) => {
                        const actualRow = task.rowStart + rIdx;
                        const actualCol = task.colStart + cIdx;
                        if (element.status === 'OK') {
                            distanceMatrix[actualRow][actualCol] = element.distance.value;
                            timeMatrix[actualRow][actualCol] = element.duration.value;
                        }
                    });
                });
                resolve();
            });
        });
    };

    try {
        await Promise.all(tasks.map(processChunk));
        return { distances: distanceMatrix, times: timeMatrix };
    } catch (err) {
        console.error("Matrix Fetch Failed", err);
        statusMessage.textContent = 'Error fetching data. Check console.';
        statusMessage.classList.add('text-red-600');
        return null;
    }
}

window.runSimulation = async function() {
    const btn = document.getElementById('runSimulationBtn');
    const statusMessage = document.getElementById('statusMessage');
    const numCustomers = customerData.length;

    if (numCustomers === 0) {
        statusMessage.textContent = 'Error: Please place at least one customer on the map.';
        statusMessage.classList.add('text-red-600');
        return;
    }

    btn.disabled = true;
    statusMessage.textContent = 'Starting simulation...';
    statusMessage.classList.add('text-yellow-600');
    statusMessage.classList.remove('text-gray-500', 'text-green-600', 'text-red-600');
    
    try {
        const locations = [depotLocation, ...customerData.map(c => ({ lat: c.lat, lng: c.lng }))];
        const matrixData = await fetchDistanceMatrix(locations);

        if (!matrixData) {
            btn.disabled = false;
            return;
        }
        
        statusMessage.textContent = `Step 2/3: Running optimization for ${numCustomers} customers...`;
        
        const vehicleType = document.getElementById('vehicleTypeSelect').value;
        const fleetConfig = { [vehicleType]: 1 };
        
        const numAnts = parseInt(document.getElementById('numAnts').value);
        const numIterations = parseInt(document.getElementById('numIterations').value);
        const alpha = parseFloat(document.getElementById('alpha').value);
        const beta = parseFloat(document.getElementById('beta').value);
        const rho = parseFloat(document.getElementById('rho').value);

        const params = { numAnts, numIterations, alpha, beta, rho, Q: 1000000 }; 
        
        currentSimulator = new AcoCvrpSimulator(depotLocation, customerData, fleetConfig, params, matrixData);
        const results = currentSimulator.optimize();

        updateResults(results); 
        await drawRoutes(currentSimulator.bestSolution, currentSimulator.customers); 

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

window.resetMap = function() {
    markers.forEach(marker => marker.setMap(null));
    directionRenderers.forEach(renderer => renderer.setMap(null)); 
    markers = [];
    directionRenderers = [];
    customerData = [];
    nextCustomerId = 1;
    currentSimulator = null;
    placeDepotMarker();
    document.getElementById('numCustomersDisplay').textContent = '0';
    handleVehicleChange(false); 
    document.getElementById('bestCost').textContent = '-';
    document.getElementById('vehiclesUsed').textContent = '-';
    document.getElementById('routeDetails').textContent = 'No routes calculated yet.';
}

function updateResults(results) {
    const solution = results.solution;
    const cost = results.cost; 
    
    const bestCostEl = document.getElementById('bestCost');
    if (bestCostEl) {
        bestCostEl.textContent = (cost / 1000).toFixed(2) + ' km';
    }

    const vehiclesUsedEl = document.getElementById('vehiclesUsed');
    if (vehiclesUsedEl) {
        const vType = selectedVehicleType.charAt(0).toUpperCase() + selectedVehicleType.slice(1);
        vehiclesUsedEl.textContent = vType;
    }

    const routeDetailsDiv = document.getElementById('routeDetails');
    
    if (solution && solution.length > 0) {
        const convergenceMsg = `<p class="mb-3 font-bold text-green-700">Converged in ${results.convergenceIteration} Iterations</p>`;
        
        const routesHtml = solution.map((route, index) => {
            if (route.length === 0) return '';
            
            const vehicle = currentSimulator.vehicles[index];
            const vType = vehicle.type.charAt(0).toUpperCase() + vehicle.type.slice(1);
            const stopCount = route.length;
            const routeStr = route.map(cIdx => `C${cIdx}`).join(' → ');
            const totalRouteCost = (currentSimulator._calculateSolutionCost([route]) / 1000).toFixed(2);
            let routeTime = 0; 
            let lastNode = 0; 
            for (const cIdx of route) {
                const customer = customerData[cIdx - 1];
                const travelTime = currentSimulator._calculateTravelTime(lastNode, cIdx, vehicle);
                const arrivalTime = routeTime + travelTime;
                routeTime = arrivalTime + customer.serviceTime; 
                lastNode = cIdx;
            }
            const travelTimeBack = currentSimulator._calculateTravelTime(lastNode, 0, vehicle);
            routeTime += travelTimeBack;
            const totalTimeString = formatDecimalHours(routeTime);

            return `
                <p class="mb-1">
                    <span class="text-indigo-600">V${vehicle.id + 1} (${vType})</span> 
                    (Stops: ${stopCount}/${vehicle.maxStops}, 
                     Time: ${totalTimeString}, 
                     Distance: ${totalRouteCost} km): 
                    Depot → ${routeStr} → Depot
                </p>
            `;
        }).join('');
        
        routeDetailsDiv.innerHTML = convergenceMsg + routesHtml;
    } else {
        routeDetailsDiv.textContent = 'Failed to find a feasible solution or no routes generated.';
    }
}

window.initMap = function() {
    map = new google.maps.Map(document.getElementById("map"), {
        center: depotLocation,
        zoom: INITIAL_MAP_ZOOM,
        mapTypeId: 'roadmap',
        streetViewControl: false,
        mapTypeControl: false,
        fullscreenControl: false
    });
    directionsService = new google.maps.DirectionsService();
    distanceMatrixService = new google.maps.DistanceMatrixService();
    placeDepotMarker();
    map.addListener('click', handleMapClick);
    document.getElementById('vehicleTypeSelect').addEventListener('change', () => handleVehicleChange(true));
    handleVehicleChange(false);
}

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

function handleMapClick(mapsMouseEvent) {
    const statusMessage = document.getElementById('statusMessage');
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
        label: { text: `C${newCustomer.id}`, color: "white", fontWeight: "bold" },
        icon: {
            path: google.maps.SymbolPath.CIRCLE,
            scale: 8,
            fillColor: colorPalette[newCustomer.id % colorPalette.length], 
            fillOpacity: 0.8,
            strokeWeight: 0
        },
        title: `C${newCustomer.id}`
    });
    markers.push(marker);
    document.getElementById('numCustomersDisplay').textContent = customerData.length;
    statusMessage.textContent = `Customer C${newCustomer.id} placed. (${customerData.length}/${maxCustomers} stops filled).`;
    statusMessage.classList.remove('text-red-600', 'text-yellow-600', 'text-green-600');
    statusMessage.classList.add('text-gray-500');
}

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

/** * --- Chunked Draw Routes ---
 * MODIFIED: Removed arrow icons (reverted to simple red line).
 * Still handles 50+ customers by batching Directions API requests.
 */
async function drawRoutes(routes, customers) {
    directionRenderers.forEach(renderer => renderer.setMap(null));
    directionRenderers = [];
    
    if (!routes || routes.length === 0) return;

    for (let vIndex = 0; vIndex < routes.length; vIndex++) {
        const route = routes[vIndex];
        if (route.length === 0) continue;
        
        const color = colorPalette[vIndex % colorPalette.length];
        
        const fullPathStops = [];
        fullPathStops.push({ lat: depotLocation.lat, lng: depotLocation.lng }); 
        route.forEach(cIdx => {
            const cust = customers[cIdx - 1];
            fullPathStops.push({ lat: cust.lat, lng: cust.lng });
        });
        fullPathStops.push({ lat: depotLocation.lat, lng: depotLocation.lng }); 

        const MAX_WAYPOINTS = 23; 
        let combinedPath = [];

        for (let i = 0; i < fullPathStops.length - 1; ) {
            const origin = fullPathStops[i];
            let remaining = (fullPathStops.length - 1) - i;
            let chunkSize = Math.min(remaining, MAX_WAYPOINTS + 1); 
            
            const destIndex = i + chunkSize;
            const destination = fullPathStops[destIndex];
            
            const waypoints = [];
            for (let j = i + 1; j < destIndex; j++) {
                waypoints.push({ location: fullPathStops[j], stopover: true });
            }

            const request = {
                origin: origin,
                destination: destination,
                waypoints: waypoints,
                travelMode: google.maps.TravelMode.DRIVING,
                optimizeWaypoints: false
            };

            try {
                const result = await new Promise((resolve, reject) => {
                    directionsService.route(request, (response, status) => {
                        if (status === 'OK') resolve(response);
                        else reject(status);
                    });
                });

                if (result.routes[0] && result.routes[0].overview_path) {
                    combinedPath = combinedPath.concat(result.routes[0].overview_path);
                }
            } catch (e) {
                console.error("Chunk direction failed", e);
            }
            i = destIndex;
        }

        // Reverted to simple line without icons
        const polyline = new google.maps.Polyline({
            path: combinedPath,
            geodesic: true,
            strokeColor: color,
            strokeOpacity: 0.8,
            strokeWeight: 5,
            map: map
        });

        directionRenderers.push(polyline);
    }
}

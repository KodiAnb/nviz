<script lang="ts">
  import { onMount } from "svelte";
  import { Network } from "$lib/Network";
  import {activation_functions} from "$lib/Activations";
  import { writable } from "svelte/store";
  import {DataFrame} from '$lib/DataFrame';
  import {hidden_nodes, feature_names,outputs} from "./Writable.js";
	import Tester from "./Tester.svelte";
	import { log, relu } from "@tensorflow/tfjs";
	import { e } from "mathjs";
  
  
  let outs;
  outputs.subscribe(value => {
    outs = value;
  }); //Gets Header names of the outputs


  export let net: any;
  $: updatePlot(net) //When the network changes the updatePlot function is called

  export let df: DataFrame;
  df = df.denormalize()

  let features = [] 
  let output_features = []
  let current_weights;
  let headers: Array<string> = Object.keys(df.data[0]); //Headers of all the features

	  for (let i: number = 0; i < headers.length; i++) {
      if(outs.includes(headers[i]) == false){ //Adds non-output features to the features array

        let feature = {}
        feature["name"] = headers[i]
        

        function get_vals(head) {
          return (row) => row[head];
        }
        const columnValues = df.data.map(get_vals(headers[i]));
        const min = Math.min(...columnValues);
        const max = Math.max(...columnValues);
        // Gets Max and Min

        feature["minimum"] = min
        feature["maximum"] = max
        feature["value"] = max
        feature["index"] = Number //Index of the feature in the net which is going to be assigned later

        features.push(feature); 

      } else{
        let feature = {}
        feature["name"] = headers[i]
        

        function get_vals(head) {
          return (row) => row[head];
        }
        const columnValues = df.data.map(get_vals(headers[i]));
        const min = Math.min(...columnValues);
        const max = Math.max(...columnValues);
        // Gets Max and Min

        feature["minimum"] = min
        feature["maximum"] = max
        feature["value"] = max
        feature["index"] = Number //Index of the feature in the net which is going to be assigned later

        output_features.push(feature); 

      }
		}
  


  let plotData = {
        source: [],
        target: [],
        value: [],
        color: []
    }

  let heatmapCount = $hidden_nodes //Number of nodes in the hidden layer

  let stats = []
  let sliced = []
  let x_axis = []
  let y_axis = []

  for (let feature of features){
    if($feature_names.includes(feature.name)){ //Makes the Graph for the heatmaps
      stats.push(feature.minimum)
      stats.push(feature.maximum)
    }
  }
  sliced.push(stats.slice(0,2))
  sliced.push(stats.slice(2))
  
 
  



  let n = new Network(net);
  let axis_names: []; //First Value is the X_Axis and the Second is the Y_Axis
  let act_func = net.trainOpts.activation
  feature_names.subscribe(value => {
    axis_names = value;
  });

  let source_indices = [];
  let extra_indices = [];
  let data = [];
  let out_data = []; 
  let input_number = net.sizes[0]


  
  function updatePlot(net){
    plotData = n.get_plot_info();
    act_func = net.trainOpts.activation
    if(data.length > 0){
      for (const nod of data){
        let id = nod.id
        let arr = []
        let highest_val;
        let smallest_val;
        let weighs = net.weights[1][id]
        let bia = net.biases[1][id]
        let update_visualizer_weights = []
        let update_weights = []
        for(let k = 0; k<weighs.length; k++){
          if(source_indices.includes(k)){
            update_visualizer_weights.push(weighs[k])
          }
          update_weights.push(weighs[k])
        }
        nod.visualizer_weights = update_visualizer_weights
        nod.weights = update_weights
        nod.bias = bia
        for(let i = 0; i<y_axis.length ; i++){
          const normal_y = normalize(y_axis[i],stats[2],stats[3])
          let row = []
          for(const x of x_axis){
            const normal_x = normalize(x,stats[0],stats[1])
            let value = normal_x*nod.visualizer_weights[0] + normal_y*nod.visualizer_weights[1] + nod.bias
            for(const feature of features){
              if(source_indices.includes(feature.index) == false){
                const normal_feature = normalize(feature.value,feature.minimum,feature.maximum)
                value += normal_feature*nod.weights[feature.index]
              }
            }

            value = activation_functions(act_func,value)

            if(x == x_axis[0] && i == 0){
              highest_val = value
              smallest_val = value
            }
            row.push(value)
            if(value > highest_val){
              highest_val = value
            } else if(value < smallest_val){
              smallest_val = value
            }
          }
          arr.push(row)
          nod.heatmap[i] = row
        }
        let h_arr = JSON.parse(JSON.stringify(nod.heatmap))
        for(let i = 0; i<h_arr.length; i++){
            for(let j = 0; j<h_arr[0].length; j++){
              let valz = h_arr[i][j]
              if(highest_val-smallest_val != 0){
                let comp_valz = (valz-smallest_val)/(highest_val-smallest_val)
                nod.visualizer_heatmap[i][j] = 2*comp_valz-1
              } else{
                nod.visualizer_heatmap[i][j] = 0
              }        
            } 
          } 
        Plotly.update(`heatmap-${nod.id}`, {}, { z: nod.visualizer_heatmap });
          
      }
    
      for(const o_nod of out_data){ 
        let holder = net.weights[2][o_nod.id]
        let new_bias = net.biases[2][o_nod.id]
        let smallest_val;
        let highest_val;
        o_nod.weights = holder
        o_nod.bias = new_bias
        let weig = o_nod.weights
        for (let y = 0; y < y_axis.length; y++){
          let row = []
          let v_row = []
            for(let x = 0; x < x_axis.length; x++){
              let value = 0
              for(const nod of data){
                let heat = nod.heatmap
                
                let id = nod.id
                value+= heat[y][x]*weig[id]
              }
              value+=o_nod.bias
              value = activation_functions(act_func,value)
              for(const feature of output_features){
                if(feature.name == o_nod.name){
                  value = denormalize(value,feature.minimum,feature.maximum)
                }
              }
              if(x == 0 && y == 0){
                highest_val = value
                smallest_val = value
              }
              row.push(value)
              if(df.types[o_nod.name] == 1){
                const val = Math.round(value);
                  
                v_row.push(df.filters[o_nod.name][val]);
              }
              if(value > highest_val){
                highest_val = value
              } else if(value < smallest_val){
                smallest_val = value
              }
            }
          o_nod.heatmap[y] = row
          o_nod.visualizer_heatmap[y] = v_row
        }
        let h_arr = JSON.parse(JSON.stringify(o_nod.heatmap))
        /*for(let i = 0; i<h_arr.length; i++){
            for(let j = 0; j<h_arr[0].length; j++){
              let valz = h_arr[i][j]
              if(df.types[o_nod.name] !== 0){
                let compvalz = (valz-smallest_val)/(highest_val-smallest_val)
                compvalz = 2*compvalz-1
                console.log(compvalz)
                compvalz *= df.filters[o_nod.name].length - 1
                compvalz = Math.round(compvalz)
                console.log(compvalz)
                o_nod.visualizer_heatmap[i][j]  = compvalz
              }
              else if(highest_val-smallest_val != 0){
                let comp_valz = (valz-smallest_val)/(highest_val-smallest_val)
                o_nod.visualizer_heatmap[i][j] = 2*comp_valz-1
              } else{
                o_nod.visualizer_heatmap[i][j] = 0
              } 
            } 
        } */
        Plotly.update(`output-${o_nod.id}`, {}, { z: o_nod.visualizer_heatmap });
      }
    }
  }
  

  

  // Reactive store subscription
  


  



  // Function to create a heatmap
  const createHeatmap = (id: string, index: number, vals: any, hovering: any, title:string, min:number, max:number) => {
    const data = [
      {
        z: vals,
        text: hovering,
        type: "heatmap",
        colorscale: "magma",
        zmin:min,
        zmax:max,
        x: x_axis,
        y:y_axis
      }
    ];

    const layout = {
      title: `${title} #${index + 1}`,
      xaxis: { 
        title: axis_names[0], 
      },
      yaxis: { 
        title: axis_names[1],
      }
    };

    Plotly.newPlot(id, data, layout);
  };

  function normalize(number, minny, maxxy){
        return (number-minny)/(maxxy-minny)

  }
  function denormalize(number, minny, maxxy){
        return (number*(maxxy-minny) + minny)

  }

  onMount(() => {
    // Render heatmaps dynamically
    plotData = n.get_plot_info();


    
    
    

      for (const ite of axis_names){   
        let ind = plotData.label.indexOf(ite) //Gets the index of the 2 chosen features from plotdata

        source_indices.push(ind)
      }


      for(let i =0;i<features.length;i++){
        features[i].index = plotData.label.indexOf(features[i].name) //Gets the index of the other features and assigns it to the feature's index property
        
      }


      for(let i = 0; i<stats.length;i+=2){
        const start = stats[i]
        const end = stats[i+1]
        const adder = (end-start)/100
        for(let l = 0; l<100;l++){
          let orig = start + adder*l
          let val = Number(orig.toFixed(2))
        
          if(i==0){
            x_axis.push(val)
          } else if(i==2){
            y_axis.push(val)
          }
          
        }
      }
      y_axis = y_axis.reverse() //Produces the numbers for the x_axis and y_axis
    

      for(let target = 0; target < heatmapCount; target++){ //Creates nodes for the hidden layer
        let node = {}
        node["id"] = target
        node["visualizer_weights"] = []
        node["weights"] = []
        node["heatmap"] = []
        node["visualizer_heatmap"] = []
        node["bias"] = net.biases[1][target]
        data.push(node)
      }
      
      for(let num = 0; num < outs.length; num++){ //Creates nodes for the output layer
        if(net.weights[2][num]){
          let node = {}
          node["name"] = outs[num]
          
          node["id"] = num
          node["weights"] = net.weights[2][num]
          node["heatmap"] = []
          node["visualizer_heatmap"] = []
          node["bias"] = net.biases[2][num]
          out_data.push(node)
        }
      }
      
      
      for (const nod of data){ //Produces the heatmap for every hidden layer node
        let id = nod.id
        
        let highest_val;
        let smallest_val; //Highest Value and Smallest Value for normalization
        let weighs = net.weights[1][id] //Grabs weights from each node from net
        nod.weights = weighs //Assigns weights to the node's weight property
        for(let k = 0; k<weighs.length; k++){ //Iterates
          if(source_indices.includes(k)){ //If the index of the weight is equal to the indexes of the visualizer weights
            nod.visualizer_weights.push(weighs[k])
          }
        }
        let act_func = net.trainOpts.activation
        let arr = [] // Temporary placement for the heatmap's values
        for(let y = 0; y < y_axis.length; y++){
          const normal_y = normalize(y_axis[y],stats[2],stats[3])
          let row = []
          for(const x of x_axis){
            const normal_x = normalize(x,stats[0],stats[1])
            let value = normal_x*nod.visualizer_weights[0] + normal_y*nod.visualizer_weights[1] + nod.bias //Gets the value from multiplyting the x and y value of each point in the graph with its weights then a bias is added on
            for(const feature of features){
              if(source_indices.includes(feature.index) == false){
                  const normal_feature = normalize(feature.value,feature.minimum,feature.maximum)
                  value += normal_feature*nod.weights[feature.index] //Adds other features times their weights
              } 
            }
            
            value = activation_functions(act_func,value)
            if(x == x_axis[0] && y == 0){ //Normalization
              highest_val = value
              smallest_val = value
            }
            row.push(value)
            if(value > highest_val){
              highest_val = value
            } else if(value < smallest_val){
              smallest_val = value
            }
          }
          arr.push(row)
      }
        
        nod.heatmap = JSON.parse(JSON.stringify(arr))//Sets the heatmap to the temporary array (actual values)
        
        let v_arr = JSON.parse(JSON.stringify(arr));

        for(let i = 0; i<v_arr.length; i++){
            for(let j = 0; j<v_arr[0].length; j++){
              let valz = v_arr[i][j]
              if(highest_val-smallest_val != 0){
                valz = (valz-smallest_val)/(highest_val-smallest_val)
                valz = 2*valz-1
                v_arr[i][j] = valz
              } else{
                v_arr[i][j] = 0
              }
            } 
          } 
        nod.visualizer_heatmap = v_arr //Sets the visualizer heatmap to the altered temporary array (what will be shown on the heatmaps)
      }

      


      for(const o_nod of out_data){ 
        let highest_val;
        let smallest_val;
        let weig = o_nod.weights
        let arr = []
        let v_arr = []
        for (let y = 0; y < y_axis.length; y++){
          let v_row = []
          let row = []
            for(let x = 0; x < x_axis.length; x++){
              let value = 0
              for(const nod of data){
                let heat = nod.heatmap
                let id = nod.id

                value+= heat[y][x]*weig[id] //Gives the sum of each point multiplied by the weight of each node

                


              }
              value+=o_nod.bias
              value = activation_functions(act_func,value)


              for(const feature of output_features){
                if(feature.name == o_nod.name){
                  value = denormalize(value,feature.minimum,feature.maximum)
                }
              }
              row.push(value)
             /* if(x == 0 && y == 0){
                highest_val = value
                smallest_val = value
              }
              
              if(value > highest_val){
                highest_val = value
              } else if(value < smallest_val){
                smallest_val = value
              }*/
             if(df.types[o_nod.name] == 1){
              const val = Math.round(value);
                
              v_row.push(df.filters[o_nod.name][val]);
             }
            }
          arr.push(row)
          v_arr.push(v_row)  
        }
        o_nod.heatmap = arr
        o_nod.visualizer_heatmap = v_arr
        
        /*let visualizer_arr = JSON.parse(JSON.stringify(o_nod.heatmap))
        for(let i = 0; i<visualizer_arr.length; i++){
            for(let j = 0; j<visualizer_arr[0].length; j++){
              let valz = visualizer_arr[i][j]
              if(df.types[o_nod.name] !== 0){
                let compvalz = (valz-smallest_val)/(highest_val-smallest_val)
                compvalz = 2*compvalz-1
                compvalz *= df.filters[o_nod.name].length - 1
                compvalz = Math.round(compvalz)
                visualizer_arr[i][j] = compvalz
              
              }
              else if(highest_val-smallest_val != 0){

                let new_valz = (valz-smallest_val)/(highest_val-smallest_val)
                new_valz = 2*new_valz-1
                visualizer_arr[i][j] = new_valz
              } else{
                visualizer_arr[i][j] = 0
              }
              
              
            } 
        } 
        o_nod.visualizer_heatmap = o_nod.heatmap*/

      
      }

      let output = {}
      
      for (let i = 0; i < heatmapCount; i++) {
        const id = `heatmap-${i}`;
        const container = document.getElementById(id);
        let z_val = data[i].visualizer_heatmap
        if (container && data[i].visualizer_heatmap) {
          
          createHeatmap(id, i, z_val,[], "Heatmap", -1,1);
          
        }
      }
      for (let j = 0; j < outs.length; j++) {
        const id = `output-${j}`;
        const container = document.getElementById(id);
        let max;
        let min;
        
        if (container && out_data[j].heatmap) {
          let z_val = out_data[j].heatmap
          let hover_data = out_data[j].visualizer_heatmap
          let name = out_data[j].name
          if(df.types[name] !== 0){
            max = df.filters[name].length - 1
            min = 0
          } else{
            var obj;

            Object.keys(output_features).forEach(x => obj = output_features[x].name === name ? output_features[x]: obj);
            max = obj.maximum
            min = obj.minimum
            
          }
          createHeatmap(id, j, z_val,hover_data, name,min,max);
            
        }
      }
  });

    /*const val = Math.round(test_results_df.data[0][header]);
    test_results[header] = df.filters[header][val];
    if (df.types[header] === 0 ) {
      test_results[header] = test_results_df.data[0][header];
    } else {
      const val = Math.round(test_results_df.data[0][header]);
      test_results[header] = df.filters[header][val];
    }*/
  
  
</script>

<!-- Heatmap Containers -->
<div style="display: grid;
            grid-template-columns: 2fr 2fr; 
            gap: 20px">
  <div id="maps-container" style= "display: grid;
              grid-template-rows: repeat({hidden_nodes}, 1fr);
              gap: 10px">
    {#each Array(heatmapCount) as _, i}
      <div id={`heatmap-${i}`} style="width: 600px; height: 400px; margin-bottom: 20px;"></div>
    {/each}
    
  </div>

  <div id="outputs-container" style= "display: grid;
  grid-template-rows: repeat({outs}, 1fr);
  gap: 10px">
  {#each Array(outs.length) as _, i}
  <div id={`output-${i}`} style="width: 600px; height: 400px; margin-bottom: 20px;"></div>
  {/each}
  </div>
  
  <div id="output" style="display: flex; 
            justify-content: center;
            align-items: center">
   
    <div>
      {#each features as feature}
        {#if source_indices.includes(feature.index) == false}
          <label for= {feature.name}>{feature.name}: {feature.value}</label>
          <input type="range" name={feature.name} bind:value={feature.value} min={Math.ceil(feature.minimum)} max={Math.ceil(feature.maximum)}/>
        {/if}
      {/each}
    </div>

  </div>
  
</div>

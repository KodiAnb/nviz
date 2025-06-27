export function activation_functions(func,value){
    if (func=="relu"){
        if(value >0){
            return value
        } else{
            return 0
        }
    } else if(func =="tanh"){
        return (Math.exp(value) - Math.exp(-value)) / (Math.exp(value) + Math.exp(-value));
    } else if(func == "sigmoid"){
        return 1 / (1 + Math.exp(-value));
    } else if(func == "leaky-relu"){
        if(value >= 0){
            return value
        } else{
            return value*.01
        }
    }
}
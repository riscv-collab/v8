console.log("run add.js")
function test(v) {
return v+1;
}
%PrepareFunctionForOptimization(test);
test(1);
%OptimizeFunctionOnNextCall(test);
test(2);

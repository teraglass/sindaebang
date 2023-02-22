const changeValue = (target) => {
  // 선택한 option의 value 값
    let targetValue = target.value;
  // console.log(target.value);
  if(targetValue != null) {
           document.getElementById("log").innerHTML
                    = targetValue
                    + " is selected";
    }
    else {
            document.getElementById("log").innerHTML
                = "You have not selected";
    }

}
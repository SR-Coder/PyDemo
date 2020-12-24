const hideDiv = (data) => {
    var eleID = document.getElementById(data).parentElement.id;
    var ele = document.getElementById(data).parentElement
    const register = document.getElementById("register")
    const login = document.getElementById("login")
    console.log(data,eleID, ele)
    if(eleID === "register"){
        if(ele.style.display === "none"){
            ele.style.display = "block"
        } else {
            ele.style.display = "none"
            login.style.display ="block"
        }
    } else if(eleID ==="login"){
        if(ele.style.display ==="none"){
            ele.style.display = "block"
        } else {
            ele.style.display ="none"
            register.style.display="block"
        }
    }
    
}
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".number-br").forEach((el) => {
    // Expressão regular para encontrar números (inteiros ou decimais)
    const regex = /\d+(\.\d+)?/g;
    el.innerHTML = el.innerHTML.replace(regex, (match) => {
      // Formatar o número encontrado
      const valor = parseFloat(match);
      if (!isNaN(valor)) {
        return valor.toLocaleString("pt-BR");
      }
      return match; // caso não seja um número válido, retorna o valor original
    });
  });
});
